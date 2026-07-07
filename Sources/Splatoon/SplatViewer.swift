import SwiftUI
import MetalKit
import QuartzCore
import simd
import MetalSplatter
import SplatIO

/// SwiftUI wrapper around a Metal view that renders a Gaussian splat PLY using
/// MetalSplatter, with a free-fly camera (drag to look, WASD to move).
struct SplatViewer: NSViewRepresentable {
    /// The splat file to display. Changing this reloads the scene.
    var url: URL?
    /// Called on the main actor as the scene starts and finishes loading.
    var onLoadingChange: (Bool) -> Void = { _ in }

    func makeCoordinator() -> SplatViewerCoordinator {
        SplatViewerCoordinator()
    }

    func makeNSView(context: Context) -> MTKView {
        let view = CameraMTKView()
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
        context.coordinator.onLoadingChange = onLoadingChange
        context.coordinator.load(url)
        // Take keyboard focus so WASD works once the view is in a window.
        DispatchQueue.main.async { [weak view] in
            view?.window?.makeFirstResponder(view)
        }
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.onLoadingChange = onLoadingChange
        context.coordinator.load(url)
    }
}

/// A Metal view that forwards mouse and keyboard input to the fly camera.
final class CameraMTKView: MTKView {
    weak var coordinator: SplatViewerCoordinator?

    override var acceptsFirstResponder: Bool { true }
    override func becomeFirstResponder() -> Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        coordinator?.look(deltaX: Float(event.deltaX), deltaY: Float(event.deltaY))
    }

    override func scrollWheel(with event: NSEvent) {
        coordinator?.dolly(Float(event.scrollingDeltaY))
    }

    override func magnify(with event: NSEvent) {
        coordinator?.dolly(Float(event.magnification) * 40)
    }

    override func keyDown(with event: NSEvent) {
        // Swallow (no system beep); repeats are ignored — we track held state.
        if !event.isARepeat { coordinator?.keyChanged(event.keyCode, pressed: true) }
    }

    override func keyUp(with event: NSEvent) {
        coordinator?.keyChanged(event.keyCode, pressed: false)
    }
}

@MainActor
final class SplatViewerCoordinator: NSObject, MTKViewDelegate {
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var renderer: SplatRenderer?

    var onLoadingChange: (Bool) -> Void = { _ in }

    private var loadedURL: URL?
    private var loadTask: Task<Void, Never>?

    private var drawableSize: CGSize = .zero

    // Fly camera. Starts at the photo's viewpoint: at the origin, looking into
    // the scene (-z after the OpenCV→OpenGL calibration below).
    private var eye = SIMD3<Float>(0, 0, 0)
    private var yaw: Float = 0     // radians, around world up (Y)
    private var pitch: Float = 0   // radians, around camera right (X)

    private var pressedKeys = Set<UInt16>()
    private var lastFrameTime: CFTimeInterval?

    private let fovy: Float = 65 * .pi / 180
    private let lookSensitivity: Float = 0.005
    private let moveSpeed: Float = 8      // world units per second

    // ANSI key codes (physical positions, layout-independent).
    private enum Key {
        static let w: UInt16 = 13, a: UInt16 = 0, s: UInt16 = 1, d: UInt16 = 2
        static let q: UInt16 = 12, e: UInt16 = 14, r: UInt16 = 15
    }

    func configure(_ view: MTKView) {
        self.device = view.device
        self.commandQueue = view.device?.makeCommandQueue()
    }

    // MARK: - Input

    func look(deltaX: Float, deltaY: Float) {
        yaw += deltaX * lookSensitivity
        pitch -= deltaY * lookSensitivity
        let limit: Float = .pi / 2 - 0.01
        pitch = min(max(pitch, -limit), limit)
    }

    func dolly(_ amount: Float) {
        eye += forwardVector * (amount * 0.05)
    }

    func keyChanged(_ code: UInt16, pressed: Bool) {
        if pressed {
            if code == Key.r { resetCamera(); return }
            pressedKeys.insert(code)
        } else {
            pressedKeys.remove(code)
        }
    }

    private func resetCamera() {
        eye = .zero
        yaw = 0
        pitch = 0
    }

    // MARK: - Camera basis

    private var forwardVector: SIMD3<Float> {
        SIMD3(cos(pitch) * sin(yaw), sin(pitch), -cos(pitch) * cos(yaw))
    }

    private func advanceCamera() {
        let now = CACurrentMediaTime()
        let dt = min(now - (lastFrameTime ?? now), 0.1)
        lastFrameTime = now
        guard !pressedKeys.isEmpty else { return }

        let forward = forwardVector
        let worldUp = SIMD3<Float>(0, 1, 0)
        let right = normalize(cross(forward, worldUp))

        var move = SIMD3<Float>(0, 0, 0)
        if pressedKeys.contains(Key.w) { move += forward }
        if pressedKeys.contains(Key.s) { move -= forward }
        if pressedKeys.contains(Key.d) { move += right }
        if pressedKeys.contains(Key.a) { move -= right }
        if pressedKeys.contains(Key.e) { move += worldUp }
        if pressedKeys.contains(Key.q) { move -= worldUp }
        if move != .zero {
            eye += normalize(move) * (moveSpeed * Float(dt))
        }
    }

    // MARK: - Loading

    func load(_ url: URL?) {
        guard url != loadedURL else { return }
        loadedURL = url
        loadTask?.cancel()

        guard let url, let device else {
            // No file yet (e.g. still generating). Leave the loading state alone
            // so the overlay is already up when a real URL arrives.
            renderer = nil
            return
        }

        loadTask = Task {
            onLoadingChange(true)
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
                // A newer load superseded this one; let it own the loading state.
                if Task.isCancelled { return }
                resetCamera()
                pressedKeys.removeAll()
                renderer = newRenderer
                onLoadingChange(false)
            } catch {
                if !Task.isCancelled {
                    print("SplatViewer failed to load \(url.lastPathComponent): \(error.localizedDescription)")
                    onLoadingChange(false)
                }
            }
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    func draw(in view: MTKView) {
        advanceCamera()

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
        let projection = perspectiveRightHand(fovy: fovy, aspect: aspect, nearZ: 0.1, farZ: 1000)

        // SHARP outputs an OpenCV-convention scene (camera looks +z, +y down).
        // Convert to OpenGL (looks -z, +y up) so the default view frames the
        // scene exactly like the input photo. This is a proper rotation
        // (π about X), so the splats are not mirrored.
        let calibration = rotation(radians: .pi, axis: SIMD3<Float>(1, 0, 0))
        let view = lookAtRightHand(eye: eye,
                                   center: eye + forwardVector,
                                   up: SIMD3<Float>(0, 1, 0)) * calibration

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

// MARK: - Matrix helpers (right-handed)

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

private func lookAtRightHand(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> matrix_float4x4 {
    let f = normalize(center - eye)
    let s = normalize(cross(f, up))
    let u = cross(s, f)
    return matrix_float4x4(columns: (
        SIMD4<Float>(s.x, u.x, -f.x, 0),
        SIMD4<Float>(s.y, u.y, -f.y, 0),
        SIMD4<Float>(s.z, u.z, -f.z, 0),
        SIMD4<Float>(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)
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
