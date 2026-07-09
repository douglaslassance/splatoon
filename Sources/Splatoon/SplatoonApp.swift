import SwiftUI
import Metal
import MetalSplatter
import SplatIO
import AVFoundation
import simd
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

@main
struct SplatoonApp: App {
    @StateObject private var meshSettings = MeshSettings()

    init() {
        if CommandLine.arguments.contains("--selftest-renderer") {
            runRendererSelfTest()
        }
        if CommandLine.arguments.contains("--selftest-mesh") {
            runMeshSelfTest()
        }
        if CommandLine.arguments.contains("--selftest-scene") {
            runSceneSelfTest()
        }
        if CommandLine.arguments.contains("--selftest-video") {
            runVideoSelfTest()
        }
        if CommandLine.arguments.contains("--selftest-render-image") {
            runRenderImageSelfTest()
        }
        if CommandLine.arguments.contains("--selftest-render-camera") {
            runRenderCameraSelfTest()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(meshSettings)
                .frame(minWidth: 860, minHeight: 600)
        }
        .windowResizability(.contentMinSize)

        Settings {
            SettingsView(settings: meshSettings)
        }
    }
}

/// Builds a mesh from a cached splat PLY headlessly, for verification.
/// Usage: `Splatoon --selftest-mesh <in.ply> <out.glb> [method] [resolution]`.
private func runMeshSelfTest() -> Never {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: "--selftest-mesh"), args.count > idx + 2 else {
        print("SELFTEST mesh: usage --selftest-mesh <in.ply> <out.glb> [method] [resolution]")
        exit(1)
    }
    let ply = URL(fileURLWithPath: args[idx + 1])
    let out = URL(fileURLWithPath: args[idx + 2])
    let method = (args.count > idx + 3 ? MeshMethod(rawValue: args[idx + 3]) : nil) ?? .grid
    let resolution = (args.count > idx + 4 ? Int(args[idx + 4]) : nil) ?? 256
    do {
        let gaussians = try SplatPLYReader.readGaussians(from: ply)
        try MeshExporter.saveGLB(gaussians: gaussians, to: out, method: method,
                                 poissonResolution: resolution)
        print("SELFTEST mesh OK [\(method.rawValue)] -> \(out.path)")
        exit(0)
    } catch {
        print("SELFTEST mesh failed: \(error)")
        exit(1)
    }
}

/// Runs the multi-image reconstruction pipeline (COLMAP + OpenSplat) headlessly
/// over a directory of images, for end-to-end verification without the GUI.
/// Usage: `Splatoon --selftest-scene <imagesDir> <out.ply> [iterations]`.
private func runSceneSelfTest() -> Never {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: "--selftest-scene"), args.count > idx + 2 else {
        print("SELFTEST scene: usage --selftest-scene <imagesDir> <out.ply> [iterations]")
        exit(1)
    }
    let imagesDir = URL(fileURLWithPath: args[idx + 1])
    let out = URL(fileURLWithPath: args[idx + 2])
    let iterations = (args.count > idx + 3 ? Int(args[idx + 3]) : nil) ?? 1000

    guard let colmap = ToolLocator.resolvedURL(for: .colmap) else {
        print("SELFTEST scene: colmap not found (set SPLATOON_COLMAP or install it)"); exit(1)
    }
    guard let trainer = ToolLocator.resolvedURL(for: .opensplat) else {
        print("SELFTEST scene: opensplat not found (set SPLATOON_OPENSPLAT or build it)"); exit(1)
    }

    let workDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("splatoon-selftest-scene", isDirectory: true)
    try? FileManager.default.removeItem(at: workDir)
    try? FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)

    let totalImages = (try? FileManager.default.contentsOfDirectory(atPath: imagesDir.path).count) ?? 0
    let reconstructor = MultiImageReconstructor(colmap: colmap, trainer: trainer)
    reconstructor.trainingIterations = iterations
    do {
        try reconstructor.run(imagesDir: imagesDir, workDir: workDir, output: out,
                              totalImages: totalImages) { update in
            print("SELFTEST scene: \(update.stageLabel) \(Int(update.fraction * 100))%")
        }
        print("SELFTEST scene OK -> \(out.path)")
        exit(0)
    } catch {
        print("SELFTEST scene failed: \(error)")
        exit(1)
    }
}

/// Runs the video path headlessly: samples frames from a video FILE (the same
/// `GalleryModel.extractFrames` the Photos path uses, just from an AVURLAsset)
/// then reconstructs. Verifies frame extraction + dense-frame reconstruction
/// without needing a Photos-library video.
/// Usage: `Splatoon --selftest-video <videoFile> <out.ply> [iterations]`.
private func runVideoSelfTest() -> Never {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: "--selftest-video"), args.count > idx + 2 else {
        print("SELFTEST video: usage --selftest-video <videoFile> <out.ply> [iterations]")
        exit(1)
    }
    let videoURL = URL(fileURLWithPath: args[idx + 1])
    let out = URL(fileURLWithPath: args[idx + 2])
    let iterations = (args.count > idx + 3 ? Int(args[idx + 3]) : nil) ?? 1000

    guard let colmap = ToolLocator.resolvedURL(for: .colmap),
          let trainer = ToolLocator.resolvedURL(for: .opensplat) else {
        print("SELFTEST video: colmap/opensplat not found"); exit(1)
    }

    let workDir = FileManager.default.temporaryDirectory.appendingPathComponent("splatoon-selftest-video", isDirectory: true)
    let framesDir = workDir.appendingPathComponent("frames", isDirectory: true)
    try? FileManager.default.removeItem(at: workDir)
    try? FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true)

    let semaphore = DispatchSemaphore(value: 0)
    var failure: String?
    Task {
        let frames = await GalleryModel.extractFrames(from: AVURLAsset(url: videoURL), to: framesDir) { done, total in
            if done == total { print("SELFTEST video: extracted \(total) frames") }
        }
        guard frames >= 3 else { failure = "extracted only \(frames) frames"; semaphore.signal(); return }
        let reconstructor = MultiImageReconstructor(colmap: colmap, trainer: trainer)
        reconstructor.trainingIterations = iterations
        do {
            try reconstructor.run(imagesDir: framesDir, workDir: workDir, output: out, totalImages: frames) { update in
                print("SELFTEST video: \(update.stageLabel) \(Int(update.fraction * 100))%")
            }
            print("SELFTEST video OK (\(frames) frames) -> \(out.path)")
        } catch {
            failure = "\(error)"
        }
        semaphore.signal()
    }
    semaphore.wait()
    if let failure { print("SELFTEST video failed: \(failure)"); exit(1) }
    exit(0)
}

/// Offscreen-renders a splat PLY to a PNG using the real MetalSplatter renderer,
/// auto-framing the point cloud's bounding box (works for both SHARP's
/// photo-viewpoint splats and COLMAP-space multi-view scenes, unlike the fly
/// camera's fixed origin-start). For visual verification without a window.
/// Usage: `Splatoon --selftest-render-image <in.ply> <out.png> [width] [height]`.
private func runRenderImageSelfTest() -> Never {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: "--selftest-render-image"), args.count > idx + 2 else {
        print("SELFTEST render-image: usage --selftest-render-image <in.ply> <out.png> [width] [height]")
        exit(1)
    }
    let ply = URL(fileURLWithPath: args[idx + 1])
    let out = URL(fileURLWithPath: args[idx + 2])
    let width = (args.count > idx + 3 ? Int(args[idx + 3]) : nil) ?? 1024
    let height = (args.count > idx + 4 ? Int(args[idx + 4]) : nil) ?? 768

    runSelfTestRender(name: "render-image", ply: ply, out: out, width: width, height: height) { points in
        // Frame the point cloud's bounding box, in the same OpenCV->OpenGL
        // flipped space the renderer's calibration maps world positions into
        // (see SplatViewer's makeViewport), so this works regardless of
        // whether eye=origin (SHARP) actually sees the scene.
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in points {
            let flipped = SIMD3<Float>(p.position.x, -p.position.y, -p.position.z)
            mn = simd_min(mn, flipped); mx = simd_max(mx, flipped)
        }
        let center = (mn + mx) / 2
        let radius = max(simd_length(mx - center), 1e-3)
        let fovy: Float = 65 * .pi / 180
        let dist = radius / max(tan(fovy / 2), 1e-3) * 1.3
        let eye = center + SIMD3<Float>(0, 0, dist)

        let calibration = rotation(radians: .pi, axis: SIMD3<Float>(1, 0, 0))
        let view = lookAtRightHand(eye: eye, center: center, up: SIMD3<Float>(0, 1, 0)) * calibration
        let projection = perspectiveRightHand(fovy: fovy, aspect: Float(width) / Float(height),
                                              nearZ: max(0.001, radius * 0.01), farZ: dist + radius * 4)
        return (view, projection)
    }
}

/// Offscreen-renders a splat PLY from one of its *actual* training camera poses
/// (parsed from OpenSplat's `cameras.json`), instead of a synthetic guess. Lets
/// you tell apart "the reconstruction is genuinely bad" from "the viewing angle
/// wasn't ever observed" — 3DGS/COLMAP scenes are only well-constrained near
/// their training viewpoints.
/// Usage: `Splatoon --selftest-render-camera <in.ply> <cameras.json> <camIndex> <out.png> [width] [height]`.
private func runRenderCameraSelfTest() -> Never {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: "--selftest-render-camera"), args.count > idx + 4 else {
        print("SELFTEST render-camera: usage --selftest-render-camera <in.ply> <cameras.json> <camIndex> <out.png> [width] [height]")
        exit(1)
    }
    let ply = URL(fileURLWithPath: args[idx + 1])
    let camerasURL = URL(fileURLWithPath: args[idx + 2])
    guard let camIndex = Int(args[idx + 3]) else {
        print("SELFTEST render-camera: camIndex must be an integer"); exit(1)
    }
    let out = URL(fileURLWithPath: args[idx + 4])

    struct Camera: Decodable {
        let position: [Float]
        let rotation: [[Float]]
        let fy: Float
        let width: Int
        let height: Int
        let img_name: String
    }
    guard let cameraData = try? Data(contentsOf: camerasURL),
          let cameras = try? JSONDecoder().decode([Camera].self, from: cameraData),
          camIndex >= 0, camIndex < cameras.count else {
        print("SELFTEST render-camera: could not read camera \(camIndex) from \(camerasURL.path)"); exit(1)
    }
    let cam = cameras[camIndex]
    print("SELFTEST render-camera: using \(cam.img_name) (index \(camIndex))")

    // Default the output raster to the camera's own aspect ratio (downscaled),
    // since fy below is calibrated against cam.height, not any output size.
    let defaultScale = max(1, cam.height / 900)
    let width = (args.count > idx + 5 ? Int(args[idx + 5]) : nil) ?? (cam.width / defaultScale)
    let height = (args.count > idx + 6 ? Int(args[idx + 6]) : nil) ?? (cam.height / defaultScale)

    runSelfTestRender(name: "render-camera", ply: ply, out: out, width: width, height: height) { _ in
        // cameras.json is camera-to-world, OpenCV convention (X right, Y down,
        // Z forward) — the same convention SplatViewer calibrates from. World ->
        // camera-OpenCV is R^T * (P - C); then the shared OpenCV->OpenGL flip.
        let c = SIMD3<Float>(cam.position[0], cam.position[1], cam.position[2])
        let r = cam.rotation
        let row0 = SIMD3<Float>(r[0][0], r[1][0], r[2][0])   // R^T's rows = R's columns
        let row1 = SIMD3<Float>(r[0][1], r[1][1], r[2][1])
        let row2 = SIMD3<Float>(r[0][2], r[1][2], r[2][2])
        let worldToCameraCV = matrix_float4x4(columns: (
            SIMD4(row0.x, row1.x, row2.x, 0), SIMD4(row0.y, row1.y, row2.y, 0),
            SIMD4(row0.z, row1.z, row2.z, 0),
            SIMD4(-dot(row0, c), -dot(row1, c), -dot(row2, c), 1)))
        let calibration = rotation(radians: .pi, axis: SIMD3<Float>(1, 0, 0))
        let view = calibration * worldToCameraCV
        let fovy = 2 * atan(Float(cam.height) / (2 * cam.fy))   // fy is calibrated against the camera's native height
        let projection = perspectiveRightHand(fovy: fovy, aspect: Float(width) / Float(height), nearZ: 0.01, farZ: 100)
        return (view, projection)
    }
}

/// Shared offscreen-render core: loads `ply`, asks `makeCamera` to build a view
/// + projection matrix from the loaded points, renders, and writes a PNG.
private func runSelfTestRender(name: String, ply: URL, out: URL, width: Int, height: Int,
                               makeCamera: @escaping ([SplatPoint]) -> (matrix_float4x4, matrix_float4x4)) -> Never {
    guard let device = MTLCreateSystemDefaultDevice(), let commandQueue = device.makeCommandQueue() else {
        print("SELFTEST \(name): no Metal device"); exit(1)
    }

    let semaphore = DispatchSemaphore(value: 0)
    var failure: String?

    Task {
        do {
            let renderer = try SplatRenderer(device: device, colorFormat: .bgra8Unorm_srgb,
                                             depthFormat: .depth32Float, sampleCount: 1,
                                             maxViewCount: 1, maxSimultaneousRenders: 3)
            let reader = try AutodetectSceneReader(ply)
            let points = try await reader.readAll()
            guard !points.isEmpty else {
                throw NSError(domain: "SelfTest", code: 1, userInfo: [NSLocalizedDescriptionKey: "PLY has no points"])
            }
            let chunk = try SplatChunk(device: device, from: points)
            _ = await renderer.addChunk(chunk)

            let (view, projection) = makeCamera(points)

            let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb,
                                                                     width: width, height: height, mipmapped: false)
            colorDesc.usage = [.renderTarget, .shaderRead]
            colorDesc.storageMode = .shared
            guard let colorTexture = device.makeTexture(descriptor: colorDesc) else {
                throw NSError(domain: "SelfTest", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not create color texture"])
            }
            let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                                     width: width, height: height, mipmapped: false)
            depthDesc.usage = .renderTarget
            depthDesc.storageMode = .private
            guard let depthTexture = device.makeTexture(descriptor: depthDesc) else {
                throw NSError(domain: "SelfTest", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not create depth texture"])
            }
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                throw NSError(domain: "SelfTest", code: 4, userInfo: [NSLocalizedDescriptionKey: "Could not create command buffer"])
            }

            let viewport = SplatRenderer.ViewportDescriptor(
                viewport: MTLViewport(originX: 0, originY: 0, width: Double(width), height: Double(height), znear: 0, zfar: 1),
                projectionMatrix: projection,
                viewMatrix: view,
                screenSize: SIMD2(x: width, y: height))

            let didRender = (try? renderer.render(
                viewports: [viewport], colorTexture: colorTexture, colorStoreAction: .store,
                depthTexture: depthTexture, rasterizationRateMap: nil, renderTargetArrayLength: 0,
                to: commandBuffer)) ?? false
            guard didRender else {
                throw NSError(domain: "SelfTest", code: 5, userInfo: [NSLocalizedDescriptionKey: "renderer.render returned false"])
            }
            commandBuffer.commit()
            await commandBuffer.completed()

            try writePNG(from: colorTexture, to: out)
            print("SELFTEST \(name) OK (\(points.count) points) -> \(out.path)")
        } catch {
            failure = "\(error)"
        }
        semaphore.signal()
    }
    semaphore.wait()
    if let failure {
        print("SELFTEST \(name) failed: \(failure)")
        exit(1)
    }
    exit(0)
}

/// Reads an RGBA8 Metal texture back and writes it as a PNG.
private func writePNG(from texture: MTLTexture, to url: URL) throws {
    let width = texture.width, height = texture.height
    let bytesPerRow = width * 4
    var pixelData = [UInt8](repeating: 0, count: bytesPerRow * height)
    texture.getBytes(&pixelData, bytesPerRow: bytesPerRow,
                     from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)

    guard let provider = CGDataProvider(data: Data(pixelData) as CFData) else {
        throw NSError(domain: "SelfTest", code: 6, userInfo: [NSLocalizedDescriptionKey: "Could not create data provider"])
    }
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    guard let cgImage = CGImage(
        width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
    ) else {
        throw NSError(domain: "SelfTest", code: 7, userInfo: [NSLocalizedDescriptionKey: "Could not create CGImage"])
    }
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw NSError(domain: "SelfTest", code: 8, userInfo: [NSLocalizedDescriptionKey: "Could not create PNG destination"])
    }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw NSError(domain: "SelfTest", code: 9, userInfo: [NSLocalizedDescriptionKey: "Could not finalize PNG"])
    }
}

/// Creates a `SplatRenderer` in the real app-bundle context to verify that
/// MetalSplatter's `Bundle.module` resolves and its `default.metallib` loads.
/// Run headlessly: `Splatoon.app/Contents/MacOS/Splatoon --selftest-renderer`.
private func runRendererSelfTest() -> Never {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("SELFTEST: no Metal device")
        exit(1)
    }
    do {
        _ = try SplatRenderer(device: device,
                              colorFormat: .bgra8Unorm_srgb,
                              depthFormat: .depth32Float,
                              sampleCount: 1,
                              maxViewCount: 1,
                              maxSimultaneousRenders: 3)
        print("SELFTEST: SplatRenderer created OK — metallib resolved via Bundle.module")
        exit(0)
    } catch {
        print("SELFTEST: SplatRenderer threw: \(error)")
        exit(1)
    }
}
