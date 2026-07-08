import SwiftUI
import SceneKit
import AppKit
import simd

/// SwiftUI wrapper around an `SCNView` that renders a `Mesh` (the same geometry
/// the .glb export writes) with flat, fullbright per-vertex colours.
///
/// Navigation matches `SplatViewer`'s free-fly camera exactly: drag to look,
/// WASD to move, Q/E up-down, scroll to dolly, R to reset. Speeds scale with the
/// mesh size so the feel is consistent for both small SHARP meshes and larger
/// multi-view scenes.
struct MeshViewer: NSViewRepresentable {
    /// The triangle mesh to display. Rebuilt only when its size changes so an
    /// unrelated SwiftUI update doesn't reset the camera.
    var mesh: Mesh

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> FlyCameraSCNView {
        let view = FlyCameraSCNView()
        view.coordinator = context.coordinator
        view.backgroundColor = NSColor(calibratedRed: 0.02, green: 0.02, blue: 0.03, alpha: 1)
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = true       // keep animating for smooth WASD movement
        view.isPlaying = true
        context.coordinator.install(mesh, in: view)
        // Take keyboard focus so WASD works once the view is in a window.
        DispatchQueue.main.async { [weak view] in view?.window?.makeFirstResponder(view) }
        return view
    }

    func updateNSView(_ view: FlyCameraSCNView, context: Context) {
        guard Self.signature(of: mesh) != context.coordinator.signature else { return }
        context.coordinator.install(mesh, in: view)
    }

    static func dismantleNSView(_ view: FlyCameraSCNView, coordinator: Coordinator) {
        coordinator.stop()
    }

    // MARK: - Geometry

    fileprivate static func geometry(from mesh: Mesh) -> SCNGeometry {
        let vertices = mesh.positions.map { SCNVector3(CGFloat($0.x), CGFloat($0.y), CGFloat($0.z)) }
        let normals = mesh.normals.map { SCNVector3(CGFloat($0.x), CGFloat($0.y), CGFloat($0.z)) }
        let vertexSource = SCNGeometrySource(vertices: vertices)
        let normalSource = SCNGeometrySource(normals: normals)

        // Per-vertex colour: float RGBA, normalised from the mesh's 0–255 bytes.
        var colorData = Data(capacity: mesh.colors.count * 4 * MemoryLayout<Float>.size)
        for c in mesh.colors {
            var rgba = SIMD4<Float>(Float(c.x), Float(c.y), Float(c.z), Float(c.w)) / 255
            withUnsafeBytes(of: &rgba) { colorData.append(contentsOf: $0) }
        }
        let colorSource = SCNGeometrySource(
            data: colorData, semantic: .color, vectorCount: mesh.colors.count,
            usesFloatComponents: true, componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: 4 * MemoryLayout<Float>.size)

        let element: SCNGeometryElement
        if mesh.indices.isEmpty {
            let points = Array(UInt32(0)..<UInt32(mesh.positions.count))
            let pointElement = SCNGeometryElement(indices: points, primitiveType: .point)
            pointElement.pointSize = 4
            pointElement.minimumPointScreenSpaceRadius = 1
            pointElement.maximumPointScreenSpaceRadius = 6
            element = pointElement
        } else {
            element = SCNGeometryElement(indices: mesh.indices, primitiveType: .triangles)
        }

        let geometry = SCNGeometry(sources: [vertexSource, normalSource, colorSource],
                                   elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant        // matches the export's KHR_materials_unlit
        material.isDoubleSided = true             // winding isn't guaranteed consistent
        material.diffuse.contents = NSColor.white // modulated by the per-vertex COLOR source
        geometry.materials = [material]
        return geometry
    }

    /// Axis-aligned bounds of the positions as (center, radius).
    fileprivate static func bounds(of positions: [SIMD3<Float>]) -> (center: SIMD3<Float>, radius: Float) {
        guard !positions.isEmpty else { return (.zero, 1) }
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in positions { mn = simd_min(mn, p); mx = simd_max(mx, p) }
        let center = (mn + mx) / 2
        return (center, max(simd_length(mx - center), 1e-3))
    }

    /// Cheap change proxy: vertex and index counts. Enough to reframe on rebuild
    /// while ignoring SwiftUI updates that leave the mesh untouched.
    fileprivate static func signature(of mesh: Mesh) -> Int {
        mesh.positions.count &* 1_000_003 &+ mesh.indices.count
    }

    // MARK: - Fly camera (mirrors SplatViewerCoordinator)

    /// Drives a free-fly camera over the SceneKit view, advanced on a main-thread
    /// timer so held keys move smoothly. All state lives on the main thread (the
    /// input handlers and the timer both fire there), so no locking is needed.
    final class Coordinator: NSObject {
        var signature: Int?

        private weak var cameraNode: SCNNode?
        private var timer: Timer?

        private var eye = SIMD3<Float>(0, 0, 0)
        private var yaw: Float = 0     // radians around world up (Y)
        private var pitch: Float = 0   // radians around camera right (X)
        private var home = (eye: SIMD3<Float>(0, 0, 0), yaw: Float(0), pitch: Float(0))

        private var pressedKeys = Set<UInt16>()
        private var lastFrameTime: CFTimeInterval?

        private let fovDegrees: Float = 65
        private let lookSensitivity: Float = 0.005
        private var moveSpeed: Float = 8        // world units/sec, scaled to mesh size
        private var dollyScale: Float = 0.05    // per scroll unit, scaled to mesh size
        private var panSensitivity: Float = 0.016   // world units per pixel, scaled to mesh size

        private enum Key {
            static let w: UInt16 = 13, a: UInt16 = 0, s: UInt16 = 1, d: UInt16 = 2
            static let q: UInt16 = 12, e: UInt16 = 14, r: UInt16 = 15
        }

        func install(_ mesh: Mesh, in view: SCNView) {
            let scene = SCNScene()
            let node = SCNNode(geometry: MeshViewer.geometry(from: mesh))
            scene.rootNode.addChildNode(node)

            let (center, radius) = MeshViewer.bounds(of: mesh.positions)
            let camera = SCNCamera()
            camera.fieldOfView = CGFloat(fovDegrees)
            camera.projectionDirection = .vertical
            camera.zNear = Double(max(0.001, radius * 0.01))
            camera.zFar = Double(radius * 40 + 10)
            let camNode = SCNNode()
            camNode.camera = camera
            scene.rootNode.addChildNode(camNode)
            cameraNode = camNode

            // Frame the mesh: back off along +Z looking -Z at its centre. Speeds
            // scale with the mesh so navigation feels the same at any size.
            let dist = radius / max(tan(fovDegrees * .pi / 180 / 2), 1e-3) * 1.3
            eye = center + SIMD3(0, 0, dist)
            yaw = 0
            pitch = 0
            home = (eye, 0, 0)
            moveSpeed = radius * 1.5
            dollyScale = radius * 0.01
            panSensitivity = moveSpeed * 0.002

            applyCameraTransform()
            view.scene = scene
            view.pointOfView = camNode
            signature = MeshViewer.signature(of: mesh)
            startTimer()
        }

        func stop() {
            timer?.invalidate()
            timer = nil
        }

        // MARK: Input (same gestures/keys as the splat viewer)

        func look(deltaX: Float, deltaY: Float) {
            yaw += deltaX * lookSensitivity
            pitch -= deltaY * lookSensitivity
            let limit: Float = .pi / 2 - 0.01
            pitch = min(max(pitch, -limit), limit)
        }

        func dolly(_ amount: Float) { eye += forwardVector * (amount * dollyScale) }

        /// Translate the camera along its screen-space right/up axes, right-drag —
        /// the mesh follows the cursor, like a grab-and-drag hand tool.
        func pan(deltaX: Float, deltaY: Float) {
            let right = normalize(cross(forwardVector, SIMD3<Float>(0, 1, 0)))
            eye -= right * (deltaX * panSensitivity)
            eye += SIMD3<Float>(0, 1, 0) * (deltaY * panSensitivity)
        }

        func keyChanged(_ code: UInt16, pressed: Bool) {
            if pressed {
                if code == Key.r { eye = home.eye; yaw = home.yaw; pitch = home.pitch; return }
                pressedKeys.insert(code)
            } else {
                pressedKeys.remove(code)
            }
        }

        // MARK: Camera update

        private var forwardVector: SIMD3<Float> {
            SIMD3(cos(pitch) * sin(yaw), sin(pitch), -cos(pitch) * cos(yaw))
        }

        private func startTimer() {
            guard timer == nil else { return }
            let timer = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
                self?.advance()
            }
            RunLoop.main.add(timer, forMode: .common)   // keep firing during scroll/resize tracking
            self.timer = timer
        }

        private func advance() {
            let now = CACurrentMediaTime()
            let dt = Float(min(now - (lastFrameTime ?? now), 0.1))
            lastFrameTime = now

            if !pressedKeys.isEmpty {
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
                if move != .zero { eye += normalize(move) * (moveSpeed * dt) }
            }
            applyCameraTransform()
        }

        private func applyCameraTransform() {
            guard let cameraNode else { return }
            let f = forwardVector
            let worldUp = SIMD3<Float>(0, 1, 0)
            let s = normalize(cross(f, worldUp))
            let u = cross(s, f)
            // Node transform is camera-to-world; SceneKit's camera looks down -Z.
            cameraNode.simdTransform = simd_float4x4(SIMD4(s, 0), SIMD4(u, 0),
                                                     SIMD4(-f, 0), SIMD4(eye, 1))
        }
    }
}

/// An `SCNView` that forwards mouse and keyboard input to the fly camera, the
/// SceneKit twin of `CameraMTKView`.
final class FlyCameraSCNView: SCNView {
    weak var coordinator: MeshViewer.Coordinator?

    override var acceptsFirstResponder: Bool { true }
    override func becomeFirstResponder() -> Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        coordinator?.look(deltaX: Float(event.deltaX), deltaY: Float(event.deltaY))
    }

    override func rightMouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func rightMouseDragged(with event: NSEvent) {
        coordinator?.pan(deltaX: Float(event.deltaX), deltaY: Float(event.deltaY))
    }

    override func scrollWheel(with event: NSEvent) {
        coordinator?.dolly(Float(event.scrollingDeltaY))
    }

    override func magnify(with event: NSEvent) {
        coordinator?.dolly(Float(event.magnification) * 40)
    }

    override func keyDown(with event: NSEvent) {
        if !event.isARepeat { coordinator?.keyChanged(event.keyCode, pressed: true) }
    }

    override func keyUp(with event: NSEvent) {
        coordinator?.keyChanged(event.keyCode, pressed: false)
    }
}
