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
    /// What to render: either an in-memory vertex-coloured `Mesh` (the on-device
    /// meshers) or a textured `.obj` bundle on disk (the OpenMVS photogrammetry
    /// mesh). Both share the same fly camera.
    enum Source {
        case mesh(Mesh)
        case obj(URL)
    }
    var source: Source
    /// Where the camera starts (and returns to on R). Identical to `SplatViewer`'s
    /// pose: the mesh lives in the same flipped space `SplatViewer`'s calibration
    /// (a π rotation about X) renders the splat into — and `MeshExport`'s `flip`
    /// is that same rotation — so the pose transfers with no conversion. nil =
    /// world origin, the SHARP single-image "eye = the photo" convention.
    var initialPose: ScenePose?

    init(mesh: Mesh, initialPose: ScenePose?) { self.source = .mesh(mesh); self.initialPose = initialPose }
    init(objURL: URL, initialPose: ScenePose?) { self.source = .obj(objURL); self.initialPose = initialPose }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> FlyCameraSCNView {
        let view = FlyCameraSCNView()
        view.coordinator = context.coordinator
        view.backgroundColor = NSColor(calibratedRed: 0.02, green: 0.02, blue: 0.03, alpha: 1)
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = true       // keep animating for smooth WASD movement
        view.isPlaying = true
        context.coordinator.install(source, initialPose: initialPose, in: view)
        // Take keyboard focus so WASD works once the view is in a window.
        DispatchQueue.main.async { [weak view] in view?.window?.makeFirstResponder(view) }
        return view
    }

    func updateNSView(_ view: FlyCameraSCNView, context: Context) {
        guard Self.signature(of: source) != context.coordinator.signature
                || initialPose != context.coordinator.installedPose else { return }
        context.coordinator.install(source, initialPose: initialPose, in: view)
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

    /// World-space bounds (center, radius) of a loaded node hierarchy, unioning
    /// every descendant geometry's box transformed by its world transform.
    fileprivate static func worldBounds(of root: SCNNode) -> (center: SIMD3<Float>, radius: Float) {
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var found = false
        func visit(_ n: SCNNode) {
            if n.geometry != nil {
                let (lo, hi) = n.boundingBox
                let m = n.simdWorldTransform
                for cx in [Float(lo.x), Float(hi.x)] {
                    for cy in [Float(lo.y), Float(hi.y)] {
                        for cz in [Float(lo.z), Float(hi.z)] {
                            let w = m * SIMD4<Float>(cx, cy, cz, 1)
                            let p = SIMD3(w.x, w.y, w.z)
                            mn = simd_min(mn, p); mx = simd_max(mx, p); found = true
                        }
                    }
                }
            }
            for c in n.childNodes { visit(c) }
        }
        visit(root)
        guard found else { return (.zero, 1) }
        let center = (mn + mx) / 2
        return (center, max(simd_length(mx - center), 1e-3))
    }

    /// A generic three-quarter framing pose for a mesh with no meaningful camera
    /// pose of its own (the OpenMVS OBJ), looking at its center from outside.
    fileprivate static func framingPose(center: SIMD3<Float>, radius: Float) -> ScenePose {
        let eye = center + simd_normalize(SIMD3<Float>(0.7, 0.4, 1.0)) * (radius * 2.4)
        let forward = simd_normalize(center - eye)
        let pitch = asin(max(-1, min(1, forward.y)))
        let yaw = atan2(forward.x, -forward.z)
        return ScenePose(eye: eye, yaw: yaw, pitch: pitch, fovyDegrees: 50)
    }

    /// Cheap change proxy so an unrelated SwiftUI update doesn't reset the camera:
    /// vertex+index counts for a `Mesh`, path+mtime for an OBJ (a rebuilt file at
    /// the same URL has a newer mtime).
    fileprivate static func signature(of source: Source) -> Int {
        switch source {
        case .mesh(let mesh):
            return mesh.positions.count &* 1_000_003 &+ mesh.indices.count
        case .obj(let url):
            let mtime = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate?.timeIntervalSince1970 ?? 0
            return url.path.hashValue &+ Int(mtime)
        }
    }

    // MARK: - Fly camera (mirrors SplatViewerCoordinator)

    /// Drives a free-fly camera over the SceneKit view, advanced on a main-thread
    /// timer so held keys move smoothly. All state lives on the main thread (the
    /// input handlers and the timer both fire there), so no locking is needed.
    final class Coordinator: NSObject {
        var signature: Int?

        private weak var cameraNode: SCNNode?
        private var timer: Timer?
        /// The pose the current scene was installed with, so `updateNSView` can
        /// detect a pose change even when the mesh geometry is unchanged.
        private(set) var installedPose: ScenePose?

        private var eye = SIMD3<Float>(0, 0, 0)
        private var yaw: Float = 0     // radians around world up (Y)
        private var pitch: Float = 0   // radians around camera right (X)
        private var home = (eye: SIMD3<Float>(0, 0, 0), yaw: Float(0), pitch: Float(0))

        private var pressedKeys = Set<UInt16>()
        private var lastFrameTime: CFTimeInterval?

        private var fovDegrees: Float = sharpFOVyDegrees   // set per-pose in install()
        private let lookSensitivity: Float = 0.0025
        private var moveSpeed: Float = 8        // world units/sec, scaled to mesh size
        private var dollyScale: Float = 0.05    // per scroll unit, scaled to mesh size
        private var panSensitivity: Float = 0.016   // world units per pixel, scaled to mesh size
        /// Scroll-wheel tuning applied on top of the size-calibrated `moveSpeed`
        /// while flying. Persists across loads so a chosen pace sticks.
        private var moveSpeedMultiplier: Float = 1

        private enum Key {
            static let w: UInt16 = 13, a: UInt16 = 0, s: UInt16 = 1, d: UInt16 = 2
            static let q: UInt16 = 12, e: UInt16 = 14, r: UInt16 = 15
        }

        func install(_ source: MeshViewer.Source, initialPose: ScenePose?, in view: SCNView) {
            let scene = SCNScene()
            let center: SIMD3<Float>
            let radius: Float
            // For the vertex-coloured `Mesh` the splat's own pose transfers exactly
            // (same flipped space), so we open at it. The OpenMVS OBJ lives in the
            // raw COLMAP frame instead (not flipped/gravity-aligned), so that pose
            // doesn't apply — frame it generically from its bounding box.
            var pose = initialPose
            switch source {
            case .mesh(let mesh):
                scene.rootNode.addChildNode(SCNNode(geometry: MeshViewer.geometry(from: mesh)))
                (center, radius) = MeshViewer.bounds(of: mesh.positions)
            case .obj(let url):
                // OpenMVS emits the mesh in COLMAP's OpenCV frame (Y down). Wrap it
                // in the same π-about-X flip the splat/mesh use so it comes in
                // roughly upright (it isn't gravity-aligned, so this is approximate).
                let flip = SCNNode()
                var t = matrix_identity_float4x4
                t.columns.1.y = -1; t.columns.2.z = -1
                flip.simdTransform = t
                if let loaded = try? SCNScene(url: url, options: [.createNormalsIfAbsent: true]) {
                    for child in loaded.rootNode.childNodes { flip.addChildNode(child) }
                    // The MTL materials are lit, but the scene has no lights (the
                    // texture is already photometrically baked), so force them
                    // fullbright — otherwise the whole mesh renders black.
                    flip.enumerateHierarchy { node, _ in
                        node.geometry?.materials.forEach { $0.lightingModel = .constant }
                    }
                }
                scene.rootNode.addChildNode(flip)
                (center, radius) = MeshViewer.worldBounds(of: scene.rootNode)
                pose = MeshViewer.framingPose(center: center, radius: radius)
            }

            // Start at the same viewpoint the splat opens from — the photo's own
            // camera — rather than a generic bounding-box angle, so toggling
            // Splat↔Mesh keeps the framing steady. nil = world origin (SHARP);
            // a registered camera pose for scenes.
            if let pose {
                eye = pose.eye; yaw = pose.yaw; pitch = pose.pitch
                fovDegrees = pose.fovyDegrees
            } else {
                eye = .zero; yaw = 0; pitch = 0
                fovDegrees = sharpFOVyDegrees
            }
            home = (eye, yaw, pitch)

            // Near/far and navigation speed still scale to the mesh's size, framed
            // from wherever the eye is (which may sit inside or beside the mesh).
            let camera = SCNCamera()
            camera.fieldOfView = CGFloat(fovDegrees)
            camera.projectionDirection = .vertical
            let eyeToFar = simd_length(center - eye) + radius
            camera.zNear = Double(max(0.001, radius * 0.002))
            camera.zFar = Double(eyeToFar * 2 + 1)
            let camNode = SCNNode()
            camNode.camera = camera
            scene.rootNode.addChildNode(camNode)
            cameraNode = camNode

            moveSpeed = radius * 0.7
            dollyScale = radius * 0.006
            panSensitivity = moveSpeed * 0.002

            applyCameraTransform()
            view.scene = scene
            view.pointOfView = camNode
            signature = MeshViewer.signature(of: source)
            installedPose = initialPose
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

        /// Scroll wheel: while a movement key is held, tune the fly speed instead
        /// of dollying — the familiar "adjust navigation sensitivity as you move"
        /// gesture. Multiplicative so one notch feels the same at any scene scale;
        /// clamped to a 0.1×…10× band around the calibrated base.
        func scroll(_ amount: Float) {
            guard !pressedKeys.isEmpty else { dolly(amount); return }
            moveSpeedMultiplier = min(max(moveSpeedMultiplier * exp(amount * 0.02), 0.1), 10)
        }

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
                if move != .zero { eye += normalize(move) * (moveSpeed * moveSpeedMultiplier * dt) }
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
    private var activityObservers: [NSObjectProtocol] = []

    override var acceptsFirstResponder: Bool { true }
    override func becomeFirstResponder() -> Bool { true }

    // Render only while the app is active (see CameraMTKView) — no reason to run
    // the SceneKit render loop while sitting in the background for hours.
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard window != nil, activityObservers.isEmpty else {
            if window == nil { removeActivityObservers() }
            return
        }
        isPlaying = NSApp.isActive
        let center = NotificationCenter.default
        activityObservers = [
            center.addObserver(forName: NSApplication.didResignActiveNotification, object: nil,
                               queue: .main) { [weak self] _ in self?.isPlaying = false },
            center.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil,
                               queue: .main) { [weak self] _ in self?.isPlaying = true },
        ]
    }

    private func removeActivityObservers() {
        activityObservers.forEach(NotificationCenter.default.removeObserver)
        activityObservers.removeAll()
    }

    deinit { removeActivityObservers() }

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
        coordinator?.scroll(Float(event.scrollingDeltaY))
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
