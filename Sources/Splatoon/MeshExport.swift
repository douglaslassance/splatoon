import Foundation
import simd

// MARK: - Mesh export (Gaussian splat -> triangle mesh)
//
// SHARP emits Gaussians on a dense image-aligned grid (empirically 768x1536).
// Two on-device strategies are offered (see MeshMethod):
//
//   .grid   - connect grid neighbours into a single 2.5D relief surface, culling
//             occlusion edges by relative depth jump. Optional Laplacian smoothing.
//   .surfel - one oriented quad per Gaussian, sized/oriented by its scale+rotation.
//             Faithful to each splat's shape, but a soup of disconnected quads.
//
// Each vertex carries position (baked OpenCV -> glTF Y-up/-Z), a normal, and a
// linear-RGB colour (glTF COLOR_0 is linear). Output is binary glTF (.glb) with a
// KHR_materials_unlit material so colours render fullbright in Unity/glTFast.

/// A triangle mesh ready to serialize (.glb) or render (SceneKit). Colours are
/// per-vertex linear RGBA bytes (glTF COLOR_0 convention); `indices` is empty for
/// a bare point cloud.
struct Mesh {
    var positions: [SIMD3<Float>]
    var normals: [SIMD3<Float>]
    var colors: [SIMD4<UInt8>]
    var indices: [UInt32]
}

enum MeshExporter {

    /// Build a triangle mesh from Gaussians, then write it as binary glTF (.glb).
    static func saveGLB(gaussians: Gaussians3D,
                        to outputURL: URL,
                        method: MeshMethod = .grid,
                        smoothGrid: Bool = false,
                        depthRatioCull: Float = 1.5,
                        surfelExtent: Float = 2.0,
                        poissonResolution: Int = 256,
                        surfaceTightness: Float = 0.5,
                        densityOffset: Float = 0.0,
                        opacityThreshold: Float = 0.05) throws {
        let mesh = buildMesh(gaussians: gaussians, method: method, smoothGrid: smoothGrid,
                             depthRatioCull: depthRatioCull, surfelExtent: surfelExtent,
                             poissonResolution: poissonResolution,
                             surfaceTightness: surfaceTightness, densityOffset: densityOffset,
                             opacityThreshold: opacityThreshold)
        try writeGLB(positions: mesh.positions, normals: mesh.normals, colors: mesh.colors,
                     indices: mesh.indices.isEmpty ? nil : mesh.indices, to: outputURL)
    }

    /// Build a triangle mesh from Gaussians using the chosen strategy. Shared by
    /// the .glb writer and the in-app SceneKit viewer so both show the same surface.
    static func buildMesh(gaussians: Gaussians3D,
                          method: MeshMethod = .grid,
                          smoothGrid: Bool = false,
                          depthRatioCull: Float = 1.5,
                          surfelExtent: Float = 2.0,
                          poissonResolution: Int = 256,
                          surfaceTightness: Float = 0.5,
                          densityOffset: Float = 0.0,
                          opacityThreshold: Float = 0.05) -> Mesh {

        let n = gaussians.count
        let meanPtr = gaussians.meanVectors.dataPointer.assumingMemoryBound(to: Float.self)
        let scalePtr = gaussians.singularValues.dataPointer.assumingMemoryBound(to: Float.self)
        let quatPtr = gaussians.quaternions.dataPointer.assumingMemoryBound(to: Float.self)
        let colorPtr = gaussians.colors.dataPointer.assumingMemoryBound(to: Float.self)
        let opacityPtr = gaussians.opacities.dataPointer.assumingMemoryBound(to: Float.self)

        // OpenCV (camera +z, +y down) -> glTF (Y-up, -Z forward).
        func flip(_ v: SIMD3<Float>) -> SIMD3<Float> { SIMD3(v.x, -v.y, -v.z) }

        func position(_ i: Int) -> SIMD3<Float> {
            flip(SIMD3(meanPtr[i * 3 + 0], meanPtr[i * 3 + 1], meanPtr[i * 3 + 2]))
        }
        func rawDepth(_ i: Int) -> Float { meanPtr[i * 3 + 2] }
        func quat(_ i: Int) -> SIMD4<Float> {
            SIMD4(quatPtr[i * 4 + 0], quatPtr[i * 4 + 1], quatPtr[i * 4 + 2], quatPtr[i * 4 + 3])
        }
        func color(_ i: Int) -> SIMD4<UInt8> {
            // glTF COLOR_0 is linear; colorPtr is already linear RGB.
            SIMD4(colorToByte(colorPtr[i * 3 + 0]),
                  colorToByte(colorPtr[i * 3 + 1]),
                  colorToByte(colorPtr[i * 3 + 2]), 255)
        }
        func colorLinear(_ i: Int) -> SIMD3<Float> {
            SIMD3(colorPtr[i * 3 + 0], colorPtr[i * 3 + 1], colorPtr[i * 3 + 2])
        }
        /// Local axis order sorted by scale descending: (major, minor, normal) with scales.
        func axes(_ i: Int) -> (major: SIMD3<Float>, minor: SIMD3<Float>, normal: SIMD3<Float>,
                                sMajor: Float, sMinor: Float) {
            let s = SIMD3(scalePtr[i * 3 + 0], scalePtr[i * 3 + 1], scalePtr[i * 3 + 2])
            let unit: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
            let order = [0, 1, 2].sorted { s[$0] > s[$1] }   // largest scale first
            let q = quat(i)
            return (flip(quatRotate(q, unit[order[0]])),
                    flip(quatRotate(q, unit[order[1]])),
                    flip(quatRotate(q, unit[order[2]])),
                    s[order[0]], s[order[1]])
        }
        func normal(_ i: Int) -> SIMD3<Float> {
            var nrm = axes(i).normal
            if nrm.z < 0 { nrm = -nrm }                       // face the camera
            let len = simd_length(nrm)
            return len > 1e-6 ? nrm / len : SIMD3(0, 0, 1)
        }
        func valid(_ i: Int) -> Bool { opacityPtr[i] > opacityThreshold }
        /// The three orthonormal local axes (in glTF space) of splat `i`.
        func axisVectors(_ i: Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
            let q = quat(i)
            return (flip(quatRotate(q, SIMD3(1, 0, 0))),
                    flip(quatRotate(q, SIMD3(0, 1, 0))),
                    flip(quatRotate(q, SIMD3(0, 0, 1))))
        }
        /// Per-axis Gaussian scale (standard deviation) of splat `i`.
        func scales(_ i: Int) -> SIMD3<Float> {
            SIMD3(scalePtr[i * 3 + 0], scalePtr[i * 3 + 1], scalePtr[i * 3 + 2])
        }
        func opacity(_ i: Int) -> Float { opacityPtr[i] }

        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var colors: [SIMD4<UInt8>] = []
        var indices: [UInt32] = []

        let gridWidth = detectGridWidth(count: n, meanPtr: meanPtr)

        switch method {
        case .grid:
            buildGridSurface(n: n, gridWidth: gridWidth, maxGridSide: 1024,
                             smooth: smoothGrid, depthRatioCull: depthRatioCull,
                             position: position, normal: normal, color: color,
                             depth: rawDepth, valid: valid,
                             positions: &positions, normals: &normals,
                             colors: &colors, indices: &indices)

        case .surfel:
            buildSurfels(n: n, gridWidth: gridWidth, maxGridSide: 640,
                         extent: surfelExtent,
                         position: position, axes: axes, normal: normal,
                         color: color, valid: valid,
                         positions: &positions, normals: &normals,
                         colors: &colors, indices: &indices)

        case .poisson:
            buildVolumetricSurface(n: n, gridWidth: gridWidth, resolution: poissonResolution,
                                   position: position, normal: normal, colorLinear: colorLinear,
                                   depth: rawDepth, valid: valid,
                                   positions: &positions, normals: &normals,
                                   colors: &colors, indices: &indices)
            if positions.isEmpty {
                // Reconstruction found no surface; fall back to the grid.
                buildGridSurface(n: n, gridWidth: gridWidth, maxGridSide: 1024,
                                 smooth: smoothGrid, depthRatioCull: depthRatioCull,
                                 position: position, normal: normal, color: color,
                                 depth: rawDepth, valid: valid,
                                 positions: &positions, normals: &normals,
                                 colors: &colors, indices: &indices)
            }

        case .density:
            buildDensitySurface(n: n, resolution: poissonResolution,
                                tightness: surfaceTightness, offset: densityOffset,
                                position: position, axisVectors: axisVectors, scales: scales,
                                opacity: opacity, colorLinear: colorLinear,
                                depth: rawDepth, valid: valid,
                                positions: &positions, normals: &normals,
                                colors: &colors, indices: &indices)
            if positions.isEmpty {
                // No surface at this iso level; fall back to the volumetric mesher.
                buildVolumetricSurface(n: n, gridWidth: gridWidth, resolution: poissonResolution,
                                       position: position, normal: normal, colorLinear: colorLinear,
                                       depth: rawDepth, valid: valid,
                                       positions: &positions, normals: &normals,
                                       colors: &colors, indices: &indices)
            }
        }

        print("MeshExporter[\(method.rawValue)]: \(n) Gaussians -> "
              + "\(positions.count) vertices, \(indices.count / 3) triangles")

        return Mesh(positions: positions, normals: normals, colors: colors, indices: indices)
    }

    // MARK: - Grid surface

    private static func buildGridSurface(
        n: Int, gridWidth: Int?, maxGridSide: Int, smooth: Bool, depthRatioCull: Float,
        position: (Int) -> SIMD3<Float>, normal: (Int) -> SIMD3<Float>,
        color: (Int) -> SIMD4<UInt8>, depth: (Int) -> Float, valid: (Int) -> Bool,
        positions: inout [SIMD3<Float>], normals: inout [SIMD3<Float>],
        colors: inout [SIMD4<UInt8>], indices: inout [UInt32]
    ) {
        guard let width = gridWidth else {
            // No grid: oriented, coloured point cloud.
            for i in 0..<n where valid(i) {
                positions.append(position(i)); normals.append(normal(i)); colors.append(color(i))
            }
            return
        }
        let height = n / width
        let strideX = max(1, Int(ceil(Double(width) / Double(maxGridSide))))
        let strideY = max(1, Int(ceil(Double(height) / Double(maxGridSide))))
        let cols = Array(Swift.stride(from: 0, to: width, by: strideX))
        let rows = Array(Swift.stride(from: 0, to: height, by: strideY))
        let gw = cols.count, gh = rows.count

        var depths = [Float](repeating: 0, count: gw * gh)
        var valids = [Bool](repeating: false, count: gw * gh)
        positions.reserveCapacity(gw * gh)
        normals.reserveCapacity(gw * gh)
        colors.reserveCapacity(gw * gh)
        for (ri, r) in rows.enumerated() {
            for (ci, c) in cols.enumerated() {
                let src = r * width + c
                positions.append(position(src))
                normals.append(normal(src))
                colors.append(color(src))
                depths[ri * gw + ci] = depth(src)
                valids[ri * gw + ci] = valid(src)
            }
        }

        if smooth {
            laplacianSmooth(&positions, gw: gw, gh: gh, valid: valids,
                            depths: depths, depthRatioCull: depthRatioCull, iterations: 3)
        }

        indices.reserveCapacity((gw - 1) * (gh - 1) * 6)
        for r in 0..<(gh - 1) {
            for c in 0..<(gw - 1) {
                let i00 = r * gw + c, i10 = i00 + 1, i01 = i00 + gw, i11 = i01 + 1
                addTriangle(i00, i10, i11, into: &indices, positions: positions,
                            normals: normals, depths: depths, valid: valids, depthRatioCull: depthRatioCull)
                addTriangle(i00, i11, i01, into: &indices, positions: positions,
                            normals: normals, depths: depths, valid: valids, depthRatioCull: depthRatioCull)
            }
        }
    }

    /// Laplacian smoothing over the grid, only averaging neighbours that are valid
    /// and not across an occlusion boundary (so edges stay crisp).
    private static func laplacianSmooth(
        _ positions: inout [SIMD3<Float>], gw: Int, gh: Int, valid: [Bool],
        depths: [Float], depthRatioCull: Float, iterations: Int
    ) {
        func continuous(_ a: Int, _ b: Int) -> Bool {
            guard valid[a], valid[b] else { return false }
            let lo = min(depths[a], depths[b]), hi = max(depths[a], depths[b])
            return lo > 1e-4 && hi <= depthRatioCull * lo
        }
        for _ in 0..<iterations {
            var next = positions
            for r in 0..<gh {
                for c in 0..<gw {
                    let i = r * gw + c
                    guard valid[i] else { continue }
                    var sum = SIMD3<Float>(0, 0, 0)
                    var count: Float = 0
                    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let nr = r + dr, nc = c + dc
                        guard nr >= 0, nr < gh, nc >= 0, nc < gw else { continue }
                        let j = nr * gw + nc
                        if continuous(i, j) { sum += positions[j]; count += 1 }
                    }
                    if count > 0 {
                        next[i] = positions[i] * 0.5 + (sum / count) * 0.5
                    }
                }
            }
            positions = next
        }
    }

    // MARK: - Surfels (one oriented quad per Gaussian)

    private static func buildSurfels(
        n: Int, gridWidth: Int?, maxGridSide: Int, extent: Float,
        position: (Int) -> SIMD3<Float>,
        axes: (Int) -> (major: SIMD3<Float>, minor: SIMD3<Float>, normal: SIMD3<Float>, sMajor: Float, sMinor: Float),
        normal: (Int) -> SIMD3<Float>, color: (Int) -> SIMD4<UInt8>, valid: (Int) -> Bool,
        positions: inout [SIMD3<Float>], normals: inout [SIMD3<Float>],
        colors: inout [SIMD4<UInt8>], indices: inout [UInt32]
    ) {
        // Pick an evenly-spaced subset (grid-strided if we know the grid) to bound size.
        var sources: [Int] = []
        if let width = gridWidth {
            let height = n / width
            let sx = max(1, Int(ceil(Double(width) / Double(maxGridSide))))
            let sy = max(1, Int(ceil(Double(height) / Double(maxGridSide))))
            for r in Swift.stride(from: 0, to: height, by: sy) {
                for c in Swift.stride(from: 0, to: width, by: sx) { sources.append(r * width + c) }
            }
        } else {
            let step = max(1, n / (maxGridSide * maxGridSide))
            sources = Array(Swift.stride(from: 0, to: n, by: step))
        }

        positions.reserveCapacity(sources.count * 4)
        for src in sources where valid(src) {
            let a = axes(src)
            let center = position(src)
            let u = a.major * (a.sMajor * extent)
            let v = a.minor * (a.sMinor * extent)
            let nrm = normal(src)
            let col = color(src)

            let base = UInt32(positions.count)
            let corners = [center - u - v, center + u - v, center + u + v, center - u + v]
            // Orient winding so the face normal agrees with the surfel normal.
            let faceNormal = cross(corners[1] - corners[0], corners[2] - corners[0])
            let flipWinding = dot(faceNormal, nrm) < 0
            for corner in corners {
                positions.append(corner); normals.append(nrm); colors.append(col)
            }
            if flipWinding {
                indices.append(contentsOf: [base, base + 2, base + 1, base, base + 3, base + 2])
            } else {
                indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
            }
        }
    }

    /// One narrow-band voxel: running weighted-average signed distance and colour.
    private struct BandVoxel {
        var tsdf: Float = 0
        var weight: Float = 0
        var color = SIMD3<Float>(repeating: 0)
        mutating func accumulate(_ sd: Float, _ col: SIMD3<Float>) {
            let nw = weight + 1
            tsdf = (tsdf * weight + sd) / nw
            color = (color * weight + col) / nw
            weight = nw
        }
    }

    // MARK: - Volumetric reconstruction (TSDF + marching tetrahedra)
    //
    // The on-device stand-in for screened Poisson: splat each oriented point's
    // signed distance (to its tangent plane) into a narrow band of a voxel grid,
    // then extract the zero level-set with marching tetrahedra. Only cells whose
    // 8 corners are all observed are meshed, so the result hugs the surface,
    // smoothed and hole-filled, without hallucinating occluded backs.

    private static func buildVolumetricSurface(
        n: Int, gridWidth: Int?, resolution: Int,
        position: (Int) -> SIMD3<Float>, normal: (Int) -> SIMD3<Float>,
        colorLinear: (Int) -> SIMD3<Float>, depth: (Int) -> Float, valid: (Int) -> Bool,
        positions: inout [SIMD3<Float>], normals: inout [SIMD3<Float>],
        colors: inout [SIMD4<UInt8>], indices: inout [UInt32]
    ) {
        // 1. Sample points and derive each normal from the local depth-grid
        //    geometry (neighbour cross-product), NOT the splat's own normal.
        //    SHARP's splats are camera-facing billboards, so their normals all
        //    point at the camera; using them makes the signed-distance field a
        //    staircase of camera-facing steps ("planes in depth"). Geometric
        //    normals follow the true surface (slanted ground, walls, ...).
        let maxSide = min(640, max(96, resolution))
        var pts0: [SIMD3<Float>] = [], nrm0: [SIMD3<Float>] = []
        var col0: [SIMD3<Float>] = [], dep0: [Float] = []

        if let width = gridWidth {
            let height = n / width
            let sx = max(1, Int(ceil(Double(width) / Double(maxSide))))
            let sy = max(1, Int(ceil(Double(height) / Double(maxSide))))
            for r in Swift.stride(from: 0, to: height, by: sy) {
                for c in Swift.stride(from: 0, to: width, by: sx) {
                    let i = r * width + c
                    guard valid(i) else { continue }
                    let cR = min(c + sx, width - 1), rD = min(r + sy, height - 1)
                    let pC = position(i)
                    var gn = cross(position(r * width + cR) - pC, position(rD * width + c) - pC)
                    if gn.z < 0 { gn = -gn }                 // face the camera
                    let len = simd_length(gn)
                    nrm0.append(len > 1e-8 ? gn / len : SIMD3(0, 0, 1))
                    pts0.append(pC); col0.append(colorLinear(i)); dep0.append(depth(i))
                }
            }
        } else {
            let step = max(1, n / (maxSide * maxSide))
            for i in Swift.stride(from: 0, to: n, by: step) where valid(i) {
                pts0.append(position(i)); nrm0.append(normal(i))
                col0.append(colorLinear(i)); dep0.append(depth(i))
            }
        }
        guard pts0.count > 100 else { return }

        // 2. Clip the far background so it doesn't inflate the bounding box (which
        //    would coarsen the voxel size and starve the near subject of detail).
        let sortedDepths = dep0.sorted()
        let medianDepth = sortedDepths[sortedDepths.count / 2]
        let p90 = sortedDepths[min(sortedDepths.count - 1, Int(Double(sortedDepths.count) * 0.90))]
        let zClip = min(p90, medianDepth * 4)

        var pts: [SIMD3<Float>] = [], nrms: [SIMD3<Float>] = [], cols: [SIMD3<Float>] = []
        var deps: [Float] = []
        for k in 0..<pts0.count where dep0[k] <= zClip {
            pts.append(pts0[k]); nrms.append(nrm0[k]); cols.append(col0[k]); deps.append(dep0[k])
        }
        guard pts.count > 100 else { return }

        // 3. Bounding box + uniform voxel size from the requested resolution.
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in pts { mn = simd_min(mn, p); mx = simd_max(mx, p) }
        let extent = mx - mn
        let longest = max(extent.x, max(extent.y, extent.z))

        // Don't resolve finer than the point cloud supports: floor the voxel at
        // ~1/3 the median point spacing. This bounds memory at very high slider
        // values (going finer just interpolates without adding real detail).
        var spacings: [Float] = []
        spacings.reserveCapacity(1000)
        var spacingRNG = SystemRandomNumberGenerator()
        for _ in 0..<min(1000, pts.count - 1) {
            let i = Int.random(in: 0..<(pts.count - 1), using: &spacingRNG)
            spacings.append(simd_distance(pts[i], pts[i + 1]))
        }
        spacings.sort()
        let medSpacing = spacings.isEmpty ? 0 : spacings[spacings.count / 2]
        let voxelFloor = max(medSpacing * 0.5, 1e-5)
        // Bound the output. On an unbounded frustum, voxels finer than the point
        // spacing explode the surface voxel count (a 50M-triangle / GB mesh), so
        // cap the effective resolution and never resolve past the data density.
        let cappedRes = min(resolution, 1024)
        let voxel = max(longest / Float(max(32, cappedRes)), voxelFloor)
        print("MeshExporter[poisson]: voxel \(voxel) (effective res \(Int(longest / voxel)))")
        let origin = mn - SIMD3<Float>(repeating: 4 * voxel)

        // 4. Sparse narrow-band splat. The band stays thin *perpendicular* to the
        //    surface (sharp), but its *lateral* reach grows with depth so the
        //    perspective-thinned far points still fill without holes.
        let trunc = voxel * 1.5                   // perpendicular half-thickness
        let lateralFactor: Float = 0.02           // lateral reach as a fraction of depth
        let maxRadius = 7
        var band = [Int64: BandVoxel](minimumCapacity: 1 << 20)

        for k in 0..<pts.count {
            let p = pts[k], nrm = nrms[k], c = cols[k]
            let reach = max(voxel * 1.5, lateralFactor * deps[k])
            let radius = min(maxRadius, Int(ceil(reach / voxel)))
            let f = (p - origin) / voxel
            let cx = Int(f.x), cy = Int(f.y), cz = Int(f.z)
            for iz in (cz - radius)...(cz + radius) {
                for iy in (cy - radius)...(cy + radius) {
                    for ix in (cx - radius)...(cx + radius) {
                        let center = origin + (SIMD3(Float(ix), Float(iy), Float(iz)) + 0.5) * voxel
                        let sd = simd_dot(center - p, nrm)
                        // Thin slab perpendicular to the surface; the box gives lateral reach.
                        if abs(sd) <= trunc {
                            band[Self.voxelKey(ix, iy, iz), default: BandVoxel()].accumulate(sd, c)
                        }
                    }
                }
            }
        }
        guard band.count > 8 else { return }
        extractZeroSurface(band: band, voxel: voxel, origin: origin,
                           positions: &positions, normals: &normals,
                           colors: &colors, indices: &indices)
    }

    // MARK: - Anisotropic density reconstruction (marching tetrahedra)
    //
    // Accumulate each Gaussian's anisotropic density (opacity · exp(-½·Mahalanobis²))
    // into a voxel grid, then extract an iso-density level-set. Unlike the TSDF path
    // this needs NO per-point normal, so it stays robust on unstructured multi-view
    // splats whose per-splat normals are unreliable. Mirrors SplatEdit's "Anisotropic
    // density" proxy: Resolution (voxel grid), Surface Tightness (iso level), Offset.

    private static func buildDensitySurface(
        n: Int, resolution: Int, tightness: Float, offset: Float,
        position: (Int) -> SIMD3<Float>,
        axisVectors: (Int) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>),
        scales: (Int) -> SIMD3<Float>, opacity: (Int) -> Float,
        colorLinear: (Int) -> SIMD3<Float>, depth: (Int) -> Float, valid: (Int) -> Bool,
        positions: inout [SIMD3<Float>], normals: inout [SIMD3<Float>],
        colors: inout [SIMD4<UInt8>], indices: inout [UInt32]
    ) {
        // 1. Subsample valid splats to a working set (bounds cost on dense clouds).
        let maxSplats = 400_000
        let step = max(1, n / maxSplats)
        var used: [Int] = []
        used.reserveCapacity(min(n, maxSplats))
        for i in Swift.stride(from: 0, to: n, by: step) where valid(i) { used.append(i) }
        guard used.count > 100 else { return }

        // 2. Clip the far background so it doesn't coarsen the voxel size.
        let sortedDepths = used.map { depth($0) }.sorted()
        let medianDepth = sortedDepths[sortedDepths.count / 2]
        let p90 = sortedDepths[min(sortedDepths.count - 1, Int(Double(sortedDepths.count) * 0.90))]
        let zClip = min(p90, medianDepth * 4)
        used = used.filter { depth($0) <= zClip }
        guard used.count > 100 else { return }

        // 3. Bounding box + uniform voxel size from the requested resolution.
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for i in used { let p = position(i); mn = simd_min(mn, p); mx = simd_max(mx, p) }
        let longest = max((mx - mn).x, max((mx - mn).y, (mx - mn).z))
        guard longest > 1e-6 else { return }
        let cappedRes = min(resolution, 1024)
        let voxel = longest / Float(max(32, cappedRes))
        let origin = mn - SIMD3<Float>(repeating: 4 * voxel)

        // 4. Splat anisotropic density into a sparse voxel grid. Each splat covers
        //    ~3σ; σ is floored at half a voxel so thin splats still register on the
        //    grid instead of falling between samples.
        let kSigma: Float = 3.0
        let maxRadius = 6
        let sigmaFloor = voxel * 0.5
        struct DensityCell {
            var density: Float = 0
            var color = SIMD3<Float>(repeating: 0)
            mutating func add(_ w: Float, _ c: SIMD3<Float>) { density += w; color += c * w }
        }
        var grid = [Int64: DensityCell](minimumCapacity: 1 << 20)

        for i in used {
            let p = position(i)
            let (a0, a1, a2) = axisVectors(i)
            let s = simd_max(scales(i), SIMD3<Float>(repeating: sigmaFloor))
            let inv = SIMD3<Float>(1, 1, 1) / s
            let alpha = min(max(opacity(i), 0), 1)
            let reach = kSigma * max(s.x, max(s.y, s.z))
            let radius = min(maxRadius, max(1, Int(ceil(reach / voxel))))
            let c = colorLinear(i)
            let f = (p - origin) / voxel
            let cx = Int(f.x), cy = Int(f.y), cz = Int(f.z)
            for iz in (cz - radius)...(cz + radius) {
                for iy in (cy - radius)...(cy + radius) {
                    for ix in (cx - radius)...(cx + radius) {
                        let center = origin + (SIMD3(Float(ix), Float(iy), Float(iz)) + 0.5) * voxel
                        let d = center - p
                        // Mahalanobis² in the splat's orthonormal local frame.
                        let u = SIMD3(simd_dot(d, a0), simd_dot(d, a1), simd_dot(d, a2)) * inv
                        let m2 = simd_dot(u, u)
                        if m2 > kSigma * kSigma { continue }
                        grid[Self.voxelKey(ix, iy, iz), default: DensityCell()].add(alpha * expf(-0.5 * m2), c)
                    }
                }
            }
        }
        guard grid.count > 8 else { return }

        // 5. Iso level relative to the field: a high percentile is the reference
        //    "solid" density; the surface sits at a fraction of it. Higher tightness
        //    pulls the surface toward denser cores; offset nudges it out (>0) or in.
        let densities = grid.values.map { $0.density }.sorted()
        let ref = densities[min(densities.count - 1, Int(Double(densities.count) * 0.99))]
        guard ref > 1e-6 else { return }
        let t = min(max(tightness, 0), 1)
        let isoFraction = 0.04 + (0.30 - 0.04) * t
        let iso = ref * isoFraction * (1 - 0.5 * min(max(offset, -1), 1))

        // 6. Convert to a signed field (iso − density: negative inside, positive
        //    outside) with density-weighted colour, then extract the level-set.
        var band = [Int64: BandVoxel](minimumCapacity: grid.count)
        for (key, cell) in grid {
            var v = BandVoxel()
            v.tsdf = iso - cell.density
            v.color = cell.density > 1e-6 ? cell.color / cell.density : SIMD3(repeating: 0)
            v.weight = 1
            band[key] = v
        }
        extractZeroSurface(band: band, voxel: voxel, origin: origin,
                           positions: &positions, normals: &normals,
                           colors: &colors, indices: &indices)
    }

    /// Bit-pack a non-negative voxel index (grid origin is below the point cloud,
    /// so indices are ≥ 0) into a single key. Shared by all voxel meshers.
    @inline(__always) static func voxelKey(_ ix: Int, _ iy: Int, _ iz: Int) -> Int64 {
        (Int64(ix) << 42) | (Int64(iy) << 21) | Int64(iz)
    }

    /// Extract the zero level-set of the scalar field held in `band[*].tsdf` with
    /// marching tetrahedra. Only cells whose 8 corners are all present are meshed,
    /// so the surface hugs the data without hallucinating unobserved regions.
    /// `outward` follows increasing field value, so a field that is negative inside
    /// the surface and positive outside yields outward-facing normals.
    private static func extractZeroSurface(
        band: [Int64: BandVoxel], voxel: Float, origin: SIMD3<Float>,
        positions: inout [SIMD3<Float>], normals: inout [SIMD3<Float>],
        colors: inout [SIMD4<UInt8>], indices: inout [UInt32]
    ) {
        // Assign dense ids for welding and O(1) corner lookup.
        let keys = Array(band.keys)
        var idOf = [Int64: Int](minimumCapacity: keys.count)
        for (i, kk) in keys.enumerated() { idOf[kk] = i }
        let voxArr = keys.map { band[$0]! }
        let voxelCount = keys.count

        // 5. Marching tetrahedra over occupied cells (all 8 corners present).
        let cornerOff: [SIMD3<Int>] = [
            SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(0, 1, 0),
            SIMD3(0, 0, 1), SIMD3(1, 0, 1), SIMD3(1, 1, 1), SIMD3(0, 1, 1),
        ]
        let tets = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                    [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]

        func cornerPos(_ ix: Int, _ iy: Int, _ iz: Int) -> SIMD3<Float> {
            origin + (SIMD3(Float(ix), Float(iy), Float(iz)) + 0.5) * voxel
        }

        // Weld crossings by their global grid edge so vertices are shared between
        // adjacent tets/cells; accumulate face normals for smooth shading.
        var weld = [Int64: UInt32](minimumCapacity: 1 << 20)
        var normalAccum: [SIMD3<Float>] = []

        func edgeVertex(_ gu: Int, _ gv: Int, _ pu: SIMD3<Float>, _ pv: SIMD3<Float>,
                        _ vu: Float, _ vv: Float, _ cu: SIMD3<Float>, _ cv: SIMD3<Float>) -> UInt32 {
            let key = gu < gv ? Int64(gu) &* Int64(voxelCount) &+ Int64(gv)
                              : Int64(gv) &* Int64(voxelCount) &+ Int64(gu)
            if let idx = weld[key] { return idx }
            let denom = vv - vu
            let s = abs(denom) < 1e-12 ? 0.5 : (0 - vu) / denom
            let idx = UInt32(positions.count)
            positions.append(pu + (pv - pu) * s)
            normals.append(.zero); normalAccum.append(.zero)
            let c = cu + (cv - cu) * s
            colors.append(SIMD4(colorToByte(c.x), colorToByte(c.y), colorToByte(c.z), 255))
            weld[key] = idx
            return idx
        }

        func addTriangle(_ a: UInt32, _ b: UInt32, _ c: UInt32, outward: SIMD3<Float>) {
            var i1 = b, i2 = c
            var faceN = cross(positions[Int(i1)] - positions[Int(a)], positions[Int(i2)] - positions[Int(a)])
            if dot(faceN, outward) < 0 { swap(&i1, &i2); faceN = -faceN }
            indices.append(contentsOf: [a, i1, i2])
            normalAccum[Int(a)] += faceN; normalAccum[Int(i1)] += faceN; normalAccum[Int(i2)] += faceN
        }

        var cval = [Float](repeating: 0, count: 8)
        var cpos = [SIMD3<Float>](repeating: .zero, count: 8)
        var ccol = [SIMD3<Float>](repeating: .zero, count: 8)
        var gidx = [Int](repeating: 0, count: 8)

        // Each occupied voxel is the min corner of a candidate cell.
        for baseKey in keys {
            let bx = Int((baseKey >> 42) & 0x1FFFFF)
            let by = Int((baseKey >> 21) & 0x1FFFFF)
            let bz = Int(baseKey & 0x1FFFFF)
            var observed = true
            for j in 0..<8 {
                let o = cornerOff[j]
                guard let cid = idOf[Self.voxelKey(bx + o.x, by + o.y, bz + o.z)] else {
                    observed = false; break
                }
                let v = voxArr[cid]
                gidx[j] = cid; cval[j] = v.tsdf; ccol[j] = v.color
                cpos[j] = cornerPos(bx + o.x, by + o.y, bz + o.z)
            }
            guard observed else { continue }

            for tet in tets {
                marchTet(tet, gidx: gidx, cval: cval, cpos: cpos, ccol: ccol,
                         edgeVertex: edgeVertex, addTriangle: addTriangle)
            }
        }

        // Normalize the accumulated smooth normals.
        for i in 0..<normals.count {
            let l = simd_length(normalAccum[i])
            normals[i] = l > 1e-9 ? normalAccum[i] / l : SIMD3(0, 0, 1)
        }
    }

    /// Triangulate one tetrahedron against the zero iso-surface, welding vertices
    /// by their global grid edge.
    private static func marchTet(
        _ t: [Int], gidx: [Int], cval: [Float], cpos: [SIMD3<Float>], ccol: [SIMD3<Float>],
        edgeVertex: (Int, Int, SIMD3<Float>, SIMD3<Float>, Float, Float, SIMD3<Float>, SIMD3<Float>) -> UInt32,
        addTriangle: (UInt32, UInt32, UInt32, SIMD3<Float>) -> Void
    ) {
        var below: [Int] = [], above: [Int] = []
        for i in t { if cval[i] < 0 { below.append(i) } else { above.append(i) } }
        if below.isEmpty || above.isEmpty { return }

        // Outward direction = toward increasing value (outside the surface).
        func mean(_ idx: [Int]) -> SIMD3<Float> {
            var s = SIMD3<Float>(0, 0, 0); for i in idx { s += cpos[i] }; return s / Float(idx.count)
        }
        let outward = mean(above) - mean(below)

        // Welded vertex on the edge from below vertex u to above vertex v.
        func vert(_ u: Int, _ v: Int) -> UInt32 {
            edgeVertex(gidx[u], gidx[v], cpos[u], cpos[v], cval[u], cval[v], ccol[u], ccol[v])
        }

        if below.count == 1 {
            addTriangle(vert(below[0], above[0]), vert(below[0], above[1]), vert(below[0], above[2]), outward)
        } else if below.count == 3 {
            addTriangle(vert(below[0], above[0]), vert(below[1], above[0]), vert(below[2], above[0]), outward)
        } else { // 2 below, 2 above -> quad -> two triangles
            let a = vert(below[0], above[0]), b = vert(below[0], above[1])
            let c = vert(below[1], above[1]), d = vert(below[1], above[0])
            addTriangle(a, b, c, outward)
            addTriangle(a, c, d, outward)
        }
    }

    // MARK: - Grid detection

    /// Find the raster width of the Gaussian grid: the divisor of `count` that
    /// minimises the vertical-neighbour distance. Returns nil if none is coherent.
    private static func detectGridWidth(count n: Int, meanPtr: UnsafeMutablePointer<Float>) -> Int? {
        func pos(_ i: Int) -> SIMD3<Float> {
            SIMD3(meanPtr[i * 3 + 0], meanPtr[i * 3 + 1], meanPtr[i * 3 + 2])
        }
        let samples = 1500
        var generator = SystemRandomNumberGenerator()

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

    /// Append a grid triangle unless it touches a dropped vertex or straddles a
    /// depth discontinuity. Winding is corrected against the vertex normals.
    private static func addTriangle(_ a: Int, _ b: Int, _ c: Int,
                                    into indices: inout [UInt32],
                                    positions: [SIMD3<Float>],
                                    normals: [SIMD3<Float>],
                                    depths: [Float],
                                    valid: [Bool],
                                    depthRatioCull: Float) {
        guard valid[a], valid[b], valid[c] else { return }
        func discontinuous(_ i: Int, _ j: Int) -> Bool {
            let lo = Swift.min(depths[i], depths[j]), hi = Swift.max(depths[i], depths[j])
            return lo <= 1e-4 || hi > depthRatioCull * lo
        }
        if discontinuous(a, b) || discontinuous(b, c) || discontinuous(c, a) { return }

        let faceNormal = cross(positions[b] - positions[a], positions[c] - positions[a])
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
        while json.count % 4 != 0 { json.append(0x20) }

        var glb = Data()
        func u32(_ v: Int) -> Data { var x = UInt32(v).littleEndian; return Data(bytes: &x, count: 4) }
        let total = 12 + 8 + json.count + 8 + bin.count

        glb.append("glTF".data(using: .ascii)!)
        glb.append(u32(2))
        glb.append(u32(total))
        glb.append(u32(json.count))
        glb.append("JSON".data(using: .ascii)!)
        glb.append(json)
        glb.append(u32(bin.count))
        glb.append(contentsOf: [0x42, 0x49, 0x4E, 0x00])  // "BIN\0"
        glb.append(bin)

        try glb.write(to: url)
    }
}
