import SwiftUI
import MetalKit
import simd
import MetalSplatter
import SplatIO

/// SwiftUI wrapper around a Metal view that renders a Gaussian splat PLY using
/// MetalSplatter, with an interactive orbit/zoom camera.
struct SplatViewer: NSViewRepresentable {
    /// The splat file to display. Changing this reloads the scene.
    var url: URL?

    func makeCoordinator() -> SplatViewerCoordinator {
        SplatViewerCoordinator()
    }

    func makeNSView(context: Context) -> MTKView {
        let view = OrbitMTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.coordinator = context.coordinator
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = 1
        view.clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.03, alpha: 1)
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        view.delegate = context.coordinator
        context.coordinator.configure(view)
        context.coordinator.load(url)
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.load(url)
    }
}

/// A Metal view that forwards mouse-drag and scroll events to the coordinator's
/// orbit camera.
final class OrbitMTKView: MTKView {
    weak var coordinator: SplatViewerCoordinator?

    override var acceptsFirstResponder: Bool { true }

    override func mouseDragged(with event: NSEvent) {
        coordinator?.orbit(deltaX: Float(event.deltaX), deltaY: Float(event.deltaY))
    }

    override func scrollWheel(with event: NSEvent) {
        coordinator?.zoom(delta: Float(event.scrollingDeltaY))
    }

    override func magnify(with event: NSEvent) {
        coordinator?.pinch(magnification: Float(event.magnification))
    }
}

@MainActor
final class SplatViewerCoordinator: NSObject, MTKViewDelegate {
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var renderer: SplatRenderer?

    private var loadedURL: URL?
    private var loadTask: Task<Void, Never>?

    private var drawableSize: CGSize = .zero

    // Orbit camera state.
    private var azimuth: Float = 0
    private var elevation: Float = 0
    private var distance: Float = 8

    private let fovy: Float = 65 * .pi / 180

    func configure(_ view: MTKView) {
        self.device = view.device
        self.commandQueue = view.device?.makeCommandQueue()
    }

    // MARK: - Camera controls

    func orbit(deltaX: Float, deltaY: Float) {
        azimuth += deltaX * 0.01
        elevation += deltaY * 0.01
        let limit: Float = .pi / 2 - 0.01
        elevation = min(max(elevation, -limit), limit)
    }

    func zoom(delta: Float) {
        distance *= (1 - delta * 0.01)
        distance = min(max(distance, 0.5), 100)
    }

    func pinch(magnification: Float) {
        distance *= (1 - magnification)
        distance = min(max(distance, 0.5), 100)
    }

    // MARK: - Loading

    func load(_ url: URL?) {
        guard url != loadedURL else { return }
        loadedURL = url
        loadTask?.cancel()

        guard let url, let device else {
            renderer = nil
            return
        }

        loadTask = Task {
            do {
                let newRenderer = try SplatRenderer(
                    device: device,
                    colorFormat: .bgra8Unorm_srgb,
                    depthFormat: .depth32Float,
                    sampleCount: 1,
                    maxViewCount: 1,
                    maxSimultaneousRenders: 3
                )
                let reader = try AutodetectSceneReader(url)
                let points = try await reader.readAll()
                let chunk = try SplatChunk(device: device, from: points)
                _ = await newRenderer.addChunk(chunk)
                if Task.isCancelled { return }
                // Reset the camera for the new scene.
                azimuth = 0
                elevation = 0
                distance = 8
                renderer = newRenderer
            } catch {
                if !Task.isCancelled {
                    print("SplatViewer failed to load \(url.lastPathComponent): \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    func draw(in view: MTKView) {
        guard let renderer, renderer.isReadyToRender,
              let commandQueue,
              let drawable = view.currentDrawable,
              drawableSize.width > 0, drawableSize.height > 0,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let viewport = makeViewport()

        let didRender = (try? renderer.render(
            viewports: [viewport],
            colorTexture: drawable.texture,
            colorStoreAction: .store,
            depthTexture: view.depthStencilTexture,
            rasterizationRateMap: nil,
            renderTargetArrayLength: 0,
            to: commandBuffer
        )) ?? false

        if didRender {
            commandBuffer.present(drawable)
        }
        commandBuffer.commit()
    }

    private func makeViewport() -> SplatRenderer.ViewportDescriptor {
        let aspect = Float(drawableSize.width / drawableSize.height)
        let projection = perspectiveRightHand(fovy: fovy, aspect: aspect, nearZ: 0.1, farZ: 100)

        // Orbit around the origin; the calibration flips common 3DGS PLYs
        // rightside-up (matches MetalSplatter's sample default).
        let calibration = rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))
        let rot = rotation(radians: elevation, axis: SIMD3<Float>(1, 0, 0))
            * rotation(radians: azimuth, axis: SIMD3<Float>(0, 1, 0))
        let view = translation(0, 0, -distance) * rot * calibration

        return SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0,
                                  width: drawableSize.width, height: drawableSize.height,
                                  znear: 0, zfar: 1),
            projectionMatrix: projection,
            viewMatrix: view,
            screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height))
        )
    }
}

// MARK: - Matrix helpers (right-handed; from Apple sample conventions)

private func rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let a = normalize(axis)
    let ct = cosf(radians), st = sinf(radians), ci = 1 - ct
    let x = a.x, y = a.y, z = a.z
    return matrix_float4x4(columns: (
        SIMD4<Float>(ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
        SIMD4<Float>(x * y * ci - z * st, ct + y * y * ci, z * y * ci + x * st, 0),
        SIMD4<Float>(x * z * ci + y * st, y * z * ci - x * st, ct + z * z * ci, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}

private func translation(_ x: Float, _ y: Float, _ z: Float) -> matrix_float4x4 {
    matrix_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(x, y, z, 1)
    ))
}

private func perspectiveRightHand(fovy: Float, aspect: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspect
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, -1),
        SIMD4<Float>(0, 0, zs * nearZ, 0)
    ))
}
