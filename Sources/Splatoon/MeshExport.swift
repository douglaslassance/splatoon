import Foundation
import simd

// MARK: - Mesh export (Gaussian splat -> triangle mesh)
//
// SHARP is a single-image, feed-forward Gaussian model: it emits Gaussians on a
// dense grid aligned to the input image (a "splatter image"). Empirically the
// grid is 768 wide x 1536 tall (N = 1,179,648) in row-major order, not square,
// so we detect the grid width at runtime rather than assuming it.
//
// Because the Gaussians are laid out like pixels, we build a triangle surface by
// connecting grid neighbours instead of running general surface reconstruction.
// Each Gaussian becomes one vertex carrying:
//   - position   (baked from the model's OpenCV convention into glTF's Y-up/-Z)
//   - normal     (the Gaussian's flattest axis: its smallest-scale local axis
//                 rotated by its quaternion)
//   - colour     (linear RGB; glTF COLOR_0 is defined as linear)
//
// Triangles that straddle an occlusion boundary are culled using a relative
// depth-jump test (scale-invariant, since scene depth spans ~2 to ~200), so the
// surface doesn't rubber-sheet across empty space.
//
// Output is binary glTF (.glb), which imports into Unity via glTFast with vertex
// colours and normals intact.

enum MeshExporter {

    /// Build a triangle mesh from the Gaussians and write it as a binary glTF.
    ///
    /// - Parameters:
    ///   - gaussians: model output.
    ///   - outputURL: destination `.glb` file.
    ///   - opacityThreshold: Gaussians below this opacity are dropped from the surface.
    ///   - depthRatioCull: a triangle is dropped if any edge's deeper endpoint is
    ///     more than this factor times the nearer one (occlusion-boundary culling).
    ///   - maxGridSide: grid rows/columns are strided down toward this size to keep
    ///     the mesh a reasonable size while preserving connectivity.
    static func saveGLB(gaussians: Gaussians3D,
                        to outputURL: URL,
                        opacityThreshold: Float = 0.05,
                        depthRatioCull: Float = 1.5,
                        maxGridSide: Int = 1024) throws {

        let n = gaussians.count
        let meanPtr = gaussians.meanVectors.dataPointer.assumingMemoryBound(to: Float.self)
        let scalePtr = gaussians.singularValues.dataPointer.assumingMemoryBound(to: Float.self)
        let quatPtr = gaussians.quaternions.dataPointer.assumingMemoryBound(to: Float.self)
        let colorPtr = gaussians.colors.dataPointer.assumingMemoryBound(to: Float.self)
        let opacityPtr = gaussians.opacities.dataPointer.assumingMemoryBound(to: Float.self)

        /// Read one Gaussian and convert it into a mesh vertex.
        /// `depth` is the raw camera-space depth (positive), used for occlusion culling.
        func vertex(_ i: Int) -> (position: SIMD3<Float>, normal: SIMD3<Float>,
                                  color: SIMD4<UInt8>, depth: Float, valid: Bool) {
            let rx = meanPtr[i * 3 + 0], ry = meanPtr[i * 3 + 1], rz = meanPtr[i * 3 + 2]
            // Position: OpenCV (camera +z, +y down) -> glTF (Y-up, -Z forward),
            // matching the π-about-X calibration the viewer applies at render time.
            let position = SIMD3<Float>(rx, -ry, -rz)

            // Normal: the flattest axis of the ellipsoid is the smallest scale.
            let s0 = scalePtr[i * 3 + 0], s1 = scalePtr[i * 3 + 1], s2 = scalePtr[i * 3 + 2]
            var axis = SIMD3<Float>(1, 0, 0)
            if s1 <= s0 && s1 <= s2 { axis = SIMD3<Float>(0, 1, 0) }
            else if s2 <= s0 && s2 <= s1 { axis = SIMD3<Float>(0, 0, 1) }
            let q = SIMD4<Float>(quatPtr[i * 4 + 0], quatPtr[i * 4 + 1],
                                 quatPtr[i * 4 + 2], quatPtr[i * 4 + 3])
            var normal = quatRotate(q, axis)            // (w, x, y, z)
            normal = SIMD3<Float>(normal.x, -normal.y, -normal.z)   // same OpenCV->glTF flip
            if normal.z < 0 { normal = -normal }        // orient toward the camera
            let length = simd_length(normal)
            normal = length > 1e-6 ? normal / length : SIMD3<Float>(0, 0, 1)

            // glTF COLOR_0 is defined as linear, and colorPtr is already linear
            // RGB, so store it directly (no sRGB encoding, which would wash out).
            let r = colorToByte(colorPtr[i * 3 + 0])
            let g = colorToByte(colorPtr[i * 3 + 1])
            let b = colorToByte(colorPtr[i * 3 + 2])

            return (position, normal, SIMD4<UInt8>(r, g, b, 255), rz, opacityPtr[i] > opacityThreshold)
        }

        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var colors: [SIMD4<UInt8>] = []
        var depths: [Float] = []
        var indices: [UInt32] = []

        let gridWidth = detectGridWidth(count: n, meanPtr: meanPtr)

        if let width = gridWidth {
            let height = n / width
            // Sample rows/columns down toward maxGridSide, preserving connectivity.
            let strideX = max(1, Int(ceil(Double(width) / Double(maxGridSide))))
            let strideY = max(1, Int(ceil(Double(height) / Double(maxGridSide))))
            let cols = Array(Swift.stride(from: 0, to: width, by: strideX))
            let rows = Array(Swift.stride(from: 0, to: height, by: strideY))
            let gw = cols.count, gh = rows.count

            positions.reserveCapacity(gw * gh)
            normals.reserveCapacity(gw * gh)
            colors.reserveCapacity(gw * gh)
            depths.reserveCapacity(gw * gh)
            var valid = [Bool](repeating: false, count: gw * gh)

            for (ri, r) in rows.enumerated() {
                for (ci, c) in cols.enumerated() {
                    let v = vertex(r * width + c)
                    positions.append(v.position)
                    normals.append(v.normal)
                    colors.append(v.color)
                    depths.append(v.depth)
                    valid[ri * gw + ci] = v.valid
                }
            }

            indices.reserveCapacity((gw - 1) * (gh - 1) * 6)
            for r in 0..<(gh - 1) {
                for c in 0..<(gw - 1) {
                    let i00 = r * gw + c
                    let i10 = i00 + 1          // +column
                    let i01 = i00 + gw         // +row
                    let i11 = i01 + 1
                    addTriangle(i00, i10, i11, into: &indices,
                                positions: positions, normals: normals, depths: depths,
                                valid: valid, depthRatioCull: depthRatioCull)
                    addTriangle(i00, i11, i01, into: &indices,
                                positions: positions, normals: normals, depths: depths,
                                valid: valid, depthRatioCull: depthRatioCull)
                }
            }
            print("MeshExporter: \(n) Gaussians, grid \(width)x\(height) "
                  + "(stride \(strideX),\(strideY) -> \(gw)x\(gh)), "
                  + "\(positions.count) vertices, \(indices.count / 3) triangles")
        } else {
            // No grid detected: fall back to an oriented, coloured point cloud so
            // the export still succeeds.
            positions.reserveCapacity(n)
            normals.reserveCapacity(n)
            colors.reserveCapacity(n)
            for i in 0..<n {
                let v = vertex(i)
                guard v.valid else { continue }
                positions.append(v.position)
                normals.append(v.normal)
                colors.append(v.color)
            }
            print("MeshExporter: \(n) Gaussians, no grid detected -> point cloud "
                  + "(\(positions.count) points)")
        }

        try writeGLB(positions: positions, normals: normals, colors: colors,
                     indices: indices.isEmpty ? nil : indices, to: outputURL)
    }

    // MARK: - Grid detection

    /// Find the raster width of the Gaussian grid. SHARP emits a dense image-like
    /// grid, but not necessarily square. The correct width `W` is the one where
    /// array element `i+W` is the spatial vertical neighbour of `i`, so we pick the
    /// divisor of `count` that minimises the vertical-neighbour distance. Returns
    /// nil if no divisor yields a coherent (roughly isotropic) grid.
    private static func detectGridWidth(count n: Int, meanPtr: UnsafeMutablePointer<Float>) -> Int? {
        func pos(_ i: Int) -> SIMD3<Float> {
            SIMD3<Float>(meanPtr[i * 3 + 0], meanPtr[i * 3 + 1], meanPtr[i * 3 + 2])
        }
        let samples = 1500
        var generator = SystemRandomNumberGenerator()

        // Horizontal baseline: distance between array-adjacent Gaussians.
        var horiz: [Float] = []
        horiz.reserveCapacity(samples)
        for _ in 0..<samples {
            let i = Int.random(in: 0..<(n - 1), using: &generator)
            horiz.append(simd_distance(pos(i), pos(i + 1)))
        }
        horiz.sort()
        let horizMedian = horiz[horiz.count / 2]

        func verticalMedian(width w: Int) -> Float {
            let h = n / w
            guard h >= 2, w >= 2 else { return .greatestFiniteMagnitude }
            var ds: [Float] = []
            ds.reserveCapacity(samples)
            for _ in 0..<samples {
                let r = Int.random(in: 0..<(h - 1), using: &generator)
                let c = Int.random(in: 0..<w, using: &generator)
                let i = r * w + c
                ds.append(simd_distance(pos(i), pos(i + w)))
            }
            ds.sort()
            return ds[ds.count / 2]
        }

        var best: (width: Int, distance: Float)?
        for w in divisors(of: n) where w >= 64 && n / w >= 64 && w <= 8192 && n / w <= 8192 {
            let d = verticalMedian(width: w)
            if best == nil || d < best!.distance { best = (w, d) }
        }
        // Accept only if the vertical spacing is comparable to the horizontal
        // spacing (a genuine grid), not an arbitrary reshape.
        guard let best, best.distance <= 2.5 * horizMedian else { return nil }
        return best.width
    }

    private static func divisors(of n: Int) -> [Int] {
        var result: [Int] = []
        var i = 1
        while i * i <= n {
            if n % i == 0 {
                result.append(i)
                if i != n / i { result.append(n / i) }
            }
            i += 1
        }
        return result
    }

    // MARK: - Geometry helpers

    /// Rotate a vector by a quaternion given as (w, x, y, z).
    private static func quatRotate(_ q: SIMD4<Float>, _ v: SIMD3<Float>) -> SIMD3<Float> {
        let u = SIMD3<Float>(q.y, q.z, q.w)  // (x, y, z)
        let w = q.x                          // scalar
        let t = 2 * cross(u, v)
        return v + w * t + cross(u, t)
    }

    private static func colorToByte(_ value: Float) -> UInt8 {
        UInt8(max(0, min(255, (value * 255).rounded())))
    }

    /// Append a triangle unless it touches a dropped vertex or straddles a depth
    /// discontinuity (an occlusion boundary). Winding is corrected so the face
    /// agrees with its vertex normals, for consistent backface culling.
    private static func addTriangle(_ a: Int, _ b: Int, _ c: Int,
                                    into indices: inout [UInt32],
                                    positions: [SIMD3<Float>],
                                    normals: [SIMD3<Float>],
                                    depths: [Float],
                                    valid: [Bool],
                                    depthRatioCull: Float) {
        guard valid[a], valid[b], valid[c] else { return }

        // Relative depth-jump test on each edge (scale-invariant occlusion cull).
        func discontinuous(_ i: Int, _ j: Int) -> Bool {
            let di = depths[i], dj = depths[j]
            let lo = Swift.min(di, dj), hi = Swift.max(di, dj)
            return lo <= 1e-4 || hi > depthRatioCull * lo
        }
        if discontinuous(a, b) || discontinuous(b, c) || discontinuous(c, a) { return }

        let pa = positions[a], pb = positions[b], pc = positions[c]
        let faceNormal = cross(pb - pa, pc - pa)
        let avgNormal = normals[a] + normals[b] + normals[c]
        if dot(faceNormal, avgNormal) < 0 {
            indices.append(contentsOf: [UInt32(a), UInt32(c), UInt32(b)])
        } else {
            indices.append(contentsOf: [UInt32(a), UInt32(b), UInt32(c)])
        }
    }

    // MARK: - glTF (.glb) writer

    private static func writeGLB(positions: [SIMD3<Float>],
                                 normals: [SIMD3<Float>],
                                 colors: [SIMD4<UInt8>],
                                 indices: [UInt32]?,
                                 to url: URL) throws {
        let vertexCount = positions.count

        // ===== Binary buffer: positions | normals | colors | (indices) =====
        var bin = Data()
        func appendFloat(_ v: Float) { var x = v; bin.append(Data(bytes: &x, count: 4)) }
        func appendUInt32(_ v: UInt32) { var x = v.littleEndian; bin.append(Data(bytes: &x, count: 4)) }

        let posOffset = bin.count
        var minP = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxP = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in positions {
            appendFloat(p.x); appendFloat(p.y); appendFloat(p.z)
            minP = simd_min(minP, p); maxP = simd_max(maxP, p)
        }
        let posLength = bin.count - posOffset

        let normOffset = bin.count
        for nrm in normals { appendFloat(nrm.x); appendFloat(nrm.y); appendFloat(nrm.z) }
        let normLength = bin.count - normOffset

        let colOffset = bin.count
        for c in colors { bin.append(contentsOf: [c.x, c.y, c.z, c.w]) }
        let colLength = bin.count - colOffset

        var idxOffset = 0, idxLength = 0, idxCount = 0
        if let indices {
            idxOffset = bin.count
            for i in indices { appendUInt32(i) }
            idxLength = bin.count - idxOffset
            idxCount = indices.count
        }
        while bin.count % 4 != 0 { bin.append(0) }

        // ===== glTF JSON =====
        var bufferViews: [[String: Any]] = [
            ["buffer": 0, "byteOffset": posOffset, "byteLength": posLength, "target": 34962],
            ["buffer": 0, "byteOffset": normOffset, "byteLength": normLength, "target": 34962],
            ["buffer": 0, "byteOffset": colOffset, "byteLength": colLength, "target": 34962],
        ]
        var accessors: [[String: Any]] = [
            ["bufferView": 0, "componentType": 5126, "count": vertexCount, "type": "VEC3",
             "min": [minP.x, minP.y, minP.z], "max": [maxP.x, maxP.y, maxP.z]],
            ["bufferView": 1, "componentType": 5126, "count": vertexCount, "type": "VEC3"],
            ["bufferView": 2, "componentType": 5121, "count": vertexCount, "type": "VEC4", "normalized": true],
        ]
        let attributes: [String: Any] = ["POSITION": 0, "NORMAL": 1, "COLOR_0": 2]
        var primitive: [String: Any] = ["attributes": attributes, "material": 0]

        if idxCount > 0 {
            bufferViews.append(["buffer": 0, "byteOffset": idxOffset, "byteLength": idxLength, "target": 34963])
            accessors.append(["bufferView": 3, "componentType": 5125, "count": idxCount, "type": "SCALAR"])
            primitive["indices"] = 3
            primitive["mode"] = 4   // triangles
        } else {
            primitive["mode"] = 0   // points
        }

        // Unlit material so the vertex colours render fullbright (unaffected by
        // scene lighting) rather than being shaded. glTF's COLOR_0 attribute can
        // only multiply base colour, so this is the in-file way to get the
        // self-lit look; true per-vertex emissive requires a custom shader.
        let material: [String: Any] = [
            "name": "SplatVertexColor",
            "pbrMetallicRoughness": [
                "baseColorFactor": [1, 1, 1, 1],
                "metallicFactor": 0,
                "roughnessFactor": 1,
            ],
            "extensions": ["KHR_materials_unlit": [String: Any]()],
        ]

        let gltf: [String: Any] = [
            "asset": ["version": "2.0", "generator": "Splatoon"],
            "extensionsUsed": ["KHR_materials_unlit"],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0, "name": "Splat"]],
            "meshes": [["primitives": [primitive]]],
            "materials": [material],
            "buffers": [["byteLength": bin.count]],
            "bufferViews": bufferViews,
            "accessors": accessors,
        ]

        var json = try JSONSerialization.data(withJSONObject: gltf, options: [])
        while json.count % 4 != 0 { json.append(0x20) }   // pad with spaces

        // ===== Assemble GLB container =====
        var glb = Data()
        func u32(_ v: Int) -> Data { var x = UInt32(v).littleEndian; return Data(bytes: &x, count: 4) }
        let total = 12 + 8 + json.count + 8 + bin.count

        glb.append("glTF".data(using: .ascii)!)   // magic
        glb.append(u32(2))                          // version
        glb.append(u32(total))                      // total length

        glb.append(u32(json.count))
        glb.append("JSON".data(using: .ascii)!)
        glb.append(json)

        glb.append(u32(bin.count))
        glb.append(contentsOf: [0x42, 0x49, 0x4E, 0x00])  // "BIN\0"
        glb.append(bin)

        try glb.write(to: url)
    }
}
