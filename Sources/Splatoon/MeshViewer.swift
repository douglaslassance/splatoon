import SwiftUI
import SceneKit
import AppKit
import simd

/// SwiftUI wrapper around an `SCNView` that renders a `Mesh` (the same geometry
/// the .glb export writes) with flat, fullbright per-vertex colours. Drag to
/// orbit, scroll to zoom (SceneKit's built-in camera control).
struct MeshViewer: NSViewRepresentable {
    /// The triangle mesh to display. Rebuilt only when its size changes so an
    /// unrelated SwiftUI update doesn't reset the user's camera.
    var mesh: Mesh

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = NSColor(calibratedRed: 0.02, green: 0.02, blue: 0.03, alpha: 1)
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = false   // unlit; colour comes from vertices
        view.antialiasingMode = .multisampling4X
        install(mesh, into: view, coordinator: context.coordinator)
        return view
    }

    func updateNSView(_ view: SCNView, context: Context) {
        let sig = Self.signature(of: mesh)
        guard sig != context.coordinator.signature else { return }
        install(mesh, into: view, coordinator: context.coordinator)
    }

    // MARK: - Scene assembly

    private func install(_ mesh: Mesh, into view: SCNView, coordinator: Coordinator) {
        let scene = SCNScene()
        let node = SCNNode(geometry: Self.geometry(from: mesh))
        scene.rootNode.addChildNode(node)

        // Frame the mesh. After the glTF Y-up/-Z flip the surface faces -Z, so a
        // camera on +Z looking back sees its front; camera control orbits from there.
        let (center, radius) = Self.bounds(of: mesh.positions)
        let camera = SCNCamera()
        camera.zNear = Double(max(0.001, radius * 0.01))
        camera.zFar = Double(radius * 20 + 10)
        let camNode = SCNNode()
        camNode.camera = camera
        let halfFOV = Float(camera.fieldOfView * .pi / 180) / 2
        let dist = radius / max(tan(halfFOV), 1e-3) * 1.3
        camNode.position = SCNVector3(CGFloat(center.x), CGFloat(center.y), CGFloat(center.z + dist))
        camNode.look(at: SCNVector3(CGFloat(center.x), CGFloat(center.y), CGFloat(center.z)))
        scene.rootNode.addChildNode(camNode)

        view.scene = scene
        view.pointOfView = camNode
        coordinator.signature = Self.signature(of: mesh)
    }

    private static func geometry(from mesh: Mesh) -> SCNGeometry {
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
    private static func bounds(of positions: [SIMD3<Float>]) -> (SIMD3<Float>, Float) {
        guard !positions.isEmpty else { return (.zero, 1) }
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in positions { mn = simd_min(mn, p); mx = simd_max(mx, p) }
        let center = (mn + mx) / 2
        return (center, max(simd_length(mx - center), 1e-3))
    }

    /// Cheap change proxy: vertex and index counts. Enough to reframe on rebuild
    /// while ignoring SwiftUI updates that leave the mesh untouched.
    private static func signature(of mesh: Mesh) -> Int {
        mesh.positions.count &* 1_000_003 &+ mesh.indices.count
    }

    final class Coordinator {
        var signature: Int?
    }
}
