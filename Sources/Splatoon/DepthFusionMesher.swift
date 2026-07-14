import Foundation
import Metal
import simd
import MetalSplatter
import SplatIO

// MARK: - Multi-view depth-fusion mesher
//
// The strong mesh pipelines (2DGS, GOF, SuGaR, MeshSplatting) don't guess a
// surface from one frozen splat cloud — they render a depth+colour map from each
// registered camera and fuse the depth maps into a watertight, vertex-coloured
// mesh (TSDF fusion). This ports that technique on-device: MetalSplatter already
// renders a high-quality alpha-blended depth buffer, and every scene splat ships
// its registered camera poses in cameras.json. We render the trained splat from
// each pose offscreen, back-project the depth into a truncated signed-distance
// band, and reuse `MeshExporter`'s marching-tetrahedra extractor to pull the
// surface out — the same welding, smooth-normal and vertex-colour code the other
// meshers use.
//
// Everything is computed in the app's glTF Y-up space (positions run through the
// same `flip(v) = (v.x, -v.y, -v.z)` the other meshers apply), so the resulting
// `Mesh` drops straight into `writeGLB` and the SceneKit preview.

enum DepthFusionMesher {

    enum FusionError: LocalizedError {
        case noDevice
        case noCameras
        case loadFailed(String)
        case renderFailed
        case emptySurface

        var errorDescription: String? {
            switch self {
            case .noDevice:          return "No Metal device available for depth fusion."
            case .noCameras:         return "This scene has no registered camera poses to fuse from."
            case .loadFailed(let m): return "Depth fusion couldn't load the splat: \(m)"
            case .renderFailed:      return "Depth fusion couldn't render the splat offscreen."
            case .emptySurface:      return "Depth fusion produced no surface."
            }
        }
    }

    /// One registered camera as stored in cameras.json (OpenSplat schema). Rotation
    /// is camera->world, row-major, OpenCV convention (X right, Y down, Z forward);
    /// position is the camera centre in the same raw world the PLY positions use.
    private struct Camera: Decodable {
        var img_name: String?
        var width: Int?
        var height: Int?
        var fx: Float?
        var fy: Float?
        var position: [Float]
        var rotation: [[Float]]
    }

    /// How many cameras a cameras.json holds without decoding it fully — used by the
    /// caller to decide whether `.fusion` is viable or should fall back.
    static func cameraCount(at url: URL) -> Int {
        guard let data = try? Data(contentsOf: url),
              let array = try? JSONSerialization.jsonObject(with: data) as? [Any] else { return 0 }
        return array.count
    }

    /// Build a fused triangle mesh from a splat PLY and its registered cameras.
    /// `resolution` is the voxel grid resolution along the scene's longest axis;
    /// `maxViews` caps how many cameras are rendered and fused (sampled evenly).
    /// Runs off the main thread (safe: `SplatRenderer` is not main-actor bound).
    static func buildMesh(plyURL: URL, camerasURL: URL,
                          resolution: Int, maxViews: Int) async throws -> Mesh {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { throw FusionError.noDevice }

        // 1. Cameras. Decode, drop malformed rows, and sample evenly to `maxViews`
        //    so a long orbit doesn't render hundreds of frames.
        guard let camData = try? Data(contentsOf: camerasURL),
              let allCameras = try? JSONDecoder().decode([Camera].self, from: camData) else {
            throw FusionError.noCameras
        }
        let usable = allCameras.filter {
            $0.position.count == 3 && $0.rotation.count == 3
                && $0.rotation.allSatisfy { $0.count == 3 }
                && ($0.fx ?? 0) > 0 && ($0.fy ?? 0) > 0
                && ($0.width ?? 0) > 0 && ($0.height ?? 0) > 0
        }
        guard usable.count >= 3 else { throw FusionError.noCameras }
        let cameras = evenlySampled(usable, keep: max(3, maxViews))

        // 2. Load the splat: points for scene scale, and a renderer to draw from.
        let points: [SplatPoint]
        do {
            let reader = try AutodetectSceneReader(plyURL)
            points = try await reader.readAll()
        } catch { throw FusionError.loadFailed(error.localizedDescription) }
        guard points.count > 16 else { throw FusionError.emptySurface }

        let renderer = try SplatRenderer(
            device: device,
            colorFormat: .bgra8Unorm,          // linear bytes (shader already delinearizes)
            depthFormat: .depth32Float,
            sampleCount: 1,
            maxViewCount: 1,
            maxSimultaneousRenders: 1,
            // Nearest-surface depth, not the alpha-weighted "expected" depth: the
            // latter averages every splat along a ray (haze + surface), landing the
            // sample between them and smearing points along the view direction. The
            // frontmost-splat depth hugs the actual surface, which is what fuses cleanly.
            highQualityDepth: false)
        do {
            let chunk = try SplatChunk(device: device, from: points)
            _ = await renderer.addChunk(chunk)
        } catch { throw FusionError.loadFailed(error.localizedDescription) }

        // 3. Scene scale in glTF space (positions flipped like the other meshers).
        //    Robust bounds via per-axis percentiles so floaters don't inflate the
        //    box and coarsen the voxel. Depth stats come from the raw positions.
        let rawPositions = points.map { $0.position }
        let gltfPositions = rawPositions.map { flip($0) }
        let (mn, mx) = occupancyBounds(gltfPositions)
        let extent = mx - mn
        let longest = max(extent.x, max(extent.y, extent.z))
        guard longest > 1e-6 else { throw FusionError.emptySurface }
        let cappedRes = min(max(resolution, 32), 1024)
        let voxel = longest / Float(cappedRes)
        let origin = mn - SIMD3<Float>(repeating: 4 * voxel)

        // 4. Render each camera offscreen and fuse its depth into the band. Bound
        //    total work by striding pixels so the deposit count stays ~constant
        //    regardless of resolution or view count.
        var band = [Int64: MeshExporter.BandVoxel](minimumCapacity: 1 << 20)
        let trunc = voxel * 2.5                 // tangent-plane band half-thickness (fills small gaps)
        let targetPoints = 3_000_000            // fused surface points across all views
        let perView = max(1, targetPoints / cameras.count)
        var texturesBySize: [String: (color: MTLTexture, depth: MTLTexture)] = [:]

        for cam in cameras {
            try Task.checkCancellation()
            // Cap the render to a moderate long edge (precision/geometry come from
            // fusion, not per-view resolution) to bound cost.
            let (W, H) = renderSize(width: cam.width!, height: cam.height!, longEdgeCap: 720)
            let key = "\(W)x\(H)"
            let tex: (color: MTLTexture, depth: MTLTexture)
            if let cached = texturesBySize[key] { tex = cached }
            else {
                guard let made = makeTextures(device: device, width: W, height: H) else {
                    throw FusionError.renderFailed
                }
                texturesBySize[key] = made
                tex = made
            }

            let C = SIMD3<Float>(cam.position[0], cam.position[1], cam.position[2])
            let r = cam.rotation
            let fwd = SIMD3<Float>(r[0][2], r[1][2], r[2][2])   // camera +Z (OpenCV forward)
            let down = SIMD3<Float>(r[0][1], r[1][1], r[2][1])  // camera +Y (OpenCV down)
            let view = lookAtRightHand(eye: C, center: C + fwd, up: -down)
            let (near, far, clip) = depthRange(cameraCenter: C, forward: fwd, positions: rawPositions)
            let proj = projection(fx: cam.fx!, fy: cam.fy!, width: W, height: H, near: near, far: far)

            let viewport = SplatRenderer.ViewportDescriptor(
                viewport: MTLViewport(originX: 0, originY: 0,
                                      width: Double(W), height: Double(H), znear: 0, zfar: 1),
                projectionMatrix: proj, viewMatrix: view, screenSize: SIMD2(W, H))

            guard try renderOnce(renderer: renderer, queue: queue, viewport: viewport,
                                 color: tex.color, depth: tex.depth) else { continue }

            fuse(color: tex.color, depth: tex.depth, width: W, height: H,
                 invViewProj: (proj * view).inverse, cameraCenterGLTF: flip(C),
                 near: near, far: far, depthClip: clip, boundsMin: mn, boundsMax: mx,
                 stride: pixelStride(width: W, height: H, target: perView),
                 voxel: voxel, origin: origin, trunc: trunc, band: &band)
        }
        guard band.count > 8 else { throw FusionError.emptySurface }

        // Drop weakly-supported voxels: a genuine surface voxel is hit by many
        // pixels across many views, while a residual depth-outlier spike is hit by
        // only a few. This prunes the last stray slivers before meshing.
        let minWeight: Float = 2
        band = band.filter { $0.value.weight >= minWeight }
        guard band.count > 8 else { throw FusionError.emptySurface }

        // 5. Extract the zero level-set with the shared marching-tetrahedra mesher.
        var positions: [SIMD3<Float>] = [], normals: [SIMD3<Float>] = []
        var colors: [SIMD4<UInt8>] = [], indices: [UInt32] = []
        MeshExporter.extractZeroSurface(band: band, voxel: voxel, origin: origin,
                                        positions: &positions, normals: &normals,
                                        colors: &colors, indices: &indices)
        guard !positions.isEmpty else { throw FusionError.emptySurface }
        var mesh = Mesh(positions: positions, normals: normals, colors: colors, indices: indices)
        // Drop small disconnected islands (stray floaters the depth still leaked),
        // keeping the substantial connected components — mirrors mesh-splatting's
        // post_process_mesh cluster filter.
        mesh = keepLargestComponents(mesh)
        print("DepthFusionMesher: \(cameras.count) views, \(band.count) band voxels -> "
              + "\(mesh.positions.count) vertices, \(mesh.indices.count / 3) triangles")
        return mesh
    }

    /// Build (or reuse) the fused mesh and write it as binary glTF (.glb).
    static func saveGLB(plyURL: URL, camerasURL: URL, to outputURL: URL,
                        resolution: Int, maxViews: Int) async throws {
        let mesh = try await buildMesh(plyURL: plyURL, camerasURL: camerasURL,
                                       resolution: resolution, maxViews: maxViews)
        try MeshExporter.saveGLB(mesh: mesh, to: outputURL)
    }

    // MARK: - Fusion (back-project depth into a TSDF band)

    /// Back-project a rendered depth map into `band`: for each confident surface
    /// pixel, deposit a truncated signed distance (to the tangent plane facing the
    /// camera) plus its fused colour into a thin slab of voxels. Multi-view
    /// averaging in `BandVoxel.accumulate` reconciles the noisy per-view depth.
    private static func fuse(color: MTLTexture, depth: MTLTexture, width: Int, height: Int,
                             invViewProj: simd_float4x4, cameraCenterGLTF: SIMD3<Float>,
                             near: Float, far: Float, depthClip: Float,
                             boundsMin: SIMD3<Float>, boundsMax: SIMD3<Float>,
                             stride: Int, voxel: Float, origin: SIMD3<Float>,
                             trunc: Float,
                             band: inout [Int64: MeshExporter.BandVoxel]) {
        let pixelCount = width * height
        var depthBuf = [Float](repeating: 0, count: pixelCount)
        var colorBuf = [UInt8](repeating: 0, count: pixelCount * 4)
        let region = MTLRegionMake2D(0, 0, width, height)
        depthBuf.withUnsafeMutableBytes {
            depth.getBytes($0.baseAddress!, bytesPerRow: width * 4, from: region, mipmapLevel: 0)
        }
        colorBuf.withUnsafeMutableBytes {
            color.getBytes($0.baseAddress!, bytesPerRow: width * 4, from: region, mipmapLevel: 0)
        }
        // 3x3 median filter: 3DGS alpha-weighted depth is salt-and-pepper noisy
        // (floaters/haze pull individual pixels off-surface), and each outlier
        // back-projects to a stray point that the mesher stretches a spike to.
        // A median over valid neighbours removes them at the source.
        medianFilterDepth(&depthBuf, width: width, height: height)

        // NDC depth -> metric camera-space z (near..far).
        func eyeZ(_ d: Float) -> Float { near * far / (far - d * (far - near)) }

        let fw = Float(width), fh = Float(height)
        // Unproject a pixel to a glTF-space point; nil if its depth is empty/clipped.
        func unproject(_ i: Int, _ j: Int) -> SIMD3<Float>? {
            guard i >= 0, i < width, j >= 0, j < height else { return nil }
            let dd = depthBuf[j * width + i]
            guard dd > 0, dd < 1 else { return nil }
            let ndc = SIMD4<Float>(2 * (Float(i) + 0.5) / fw - 1,
                                   1 - 2 * (Float(j) + 0.5) / fh, dd, 1)
            let hw = invViewProj * ndc
            guard abs(hw.w) > 1e-9 else { return nil }
            return SIMD3<Float>(hw.x, hw.y, hw.z) / hw.w
        }

        let radius = max(1, Int((trunc / voxel).rounded()))
        for j in Swift.stride(from: 0, to: height, by: stride) {
            for i in Swift.stride(from: 0, to: width, by: stride) {
                let d = depthBuf[j * width + i]
                if d <= 0 || d >= 1 { continue }            // 0 = empty (background), 1 = far clip
                let z = eyeZ(d)
                if z > depthClip { continue }               // far background / floaters
                // Skip occlusion edges: a large depth jump to a neighbour means this
                // pixel straddles a silhouette, where the estimated normal is
                // meaningless and fusing bridges foreground to background.
                if isDepthEdge(depthBuf, width, height, i, j, stride, z, eyeZ) { continue }

                // bgra8Unorm, premultiplied alpha: surface colour = rgb / a.
                let ci = (j * width + i) * 4
                let a = Float(colorBuf[ci + 3]) / 255
                if a < 0.5 { continue }                     // low coverage: not a confident surface
                let inv = 1 / a
                let col = SIMD3<Float>(Float(colorBuf[ci + 2]) / 255 * inv,   // R (bgra -> index 2)
                                       Float(colorBuf[ci + 1]) / 255 * inv,   // G
                                       Float(colorBuf[ci + 0]) / 255 * inv)   // B

                guard let p = unproject(i, j),
                      let pR = unproject(i + stride, j),
                      let pD = unproject(i, j + stride) else { continue }
                if any(p .< boundsMin) || any(p .> boundsMax) { continue }

                // Geometric surface normal from the neighbour back-projections (not
                // the view direction), so the deposited band hugs the true surface
                // even at grazing angles instead of smearing along the ray. Orient
                // it toward the camera (the side it was seen from).
                var nrm = simd_cross(pR - p, pD - p)
                let nl = simd_length(nrm)
                if nl < 1e-12 { continue }
                nrm /= nl
                if simd_dot(nrm, cameraCenterGLTF - p) < 0 { nrm = -nrm }

                // Deposit the signed distance to the tangent plane through p into a
                // thin slab of voxels. sd > 0 on the camera side (outside), < 0
                // behind (inside) — matching `extractZeroSurface`'s convention.
                // Many views average in `BandVoxel`, cancelling per-view depth noise.
                let f = (p - origin) / voxel
                let cx = Int(f.x), cy = Int(f.y), cz = Int(f.z)
                for iz in (cz - radius)...(cz + radius) {
                    for iy in (cy - radius)...(cy + radius) {
                        for ix in (cx - radius)...(cx + radius) {
                            if ix < 0 || iy < 0 || iz < 0 { continue }
                            let center = origin + (SIMD3(Float(ix), Float(iy), Float(iz)) + 0.5) * voxel
                            let sd = simd_dot(center - p, nrm)
                            if abs(sd) <= trunc {
                                band[MeshExporter.voxelKey(ix, iy, iz), default: MeshExporter.BandVoxel()]
                                    .accumulate(sd, col)
                            }
                        }
                    }
                }
            }
        }
    }

    /// Replace each depth sample with the median of its valid 3x3 neighbours,
    /// removing isolated outliers (empty samples, coded 0, are excluded and left 0).
    private static func medianFilterDepth(_ buf: inout [Float], width: Int, height: Int) {
        let src = buf
        var window: [Float] = []
        window.reserveCapacity(9)
        for j in 0..<height {
            for i in 0..<width {
                let c = src[j * width + i]
                if c <= 0 || c >= 1 { continue }              // leave empty/clip pixels as-is
                window.removeAll(keepingCapacity: true)
                for dj in -1...1 {
                    let nj = j + dj
                    if nj < 0 || nj >= height { continue }
                    for di in -1...1 {
                        let ni = i + di
                        if ni < 0 || ni >= width { continue }
                        let v = src[nj * width + ni]
                        if v > 0 && v < 1 { window.append(v) }
                    }
                }
                if window.count >= 5 { window.sort(); buf[j * width + i] = window[window.count / 2] }
            }
        }
    }

    /// Whether pixel (i,j) sits on a depth discontinuity: its metric depth jumps by
    /// more than ~30% relative to its right/down neighbour `stride` pixels away.
    private static func isDepthEdge(_ depthBuf: [Float], _ width: Int, _ height: Int,
                                    _ i: Int, _ j: Int, _ stride: Int, _ z: Float,
                                    _ eyeZ: (Float) -> Float) -> Bool {
        for (di, dj) in [(stride, 0), (0, stride)] {
            let ni = i + di, nj = j + dj
            guard ni < width, nj < height else { continue }
            let dn = depthBuf[nj * width + ni]
            guard dn > 0, dn < 1 else { continue }
            let zn = eyeZ(dn)
            let lo = min(z, zn), hi = max(z, zn)
            if lo > 1e-5 && hi > 1.3 * lo { return true }
        }
        return false
    }

    /// Keep only connected components with a meaningful triangle count, dropping
    /// the small floating islands the fused depth still leaves behind. Vertices are
    /// welded by `extractZeroSurface`, so shared indices define connectivity
    /// (union-find over triangle edges). Returns the mesh unchanged if it would
    /// remove everything.
    private static func keepLargestComponents(_ mesh: Mesh) -> Mesh {
        let vcount = mesh.positions.count
        guard vcount > 0, !mesh.indices.isEmpty else { return mesh }

        var parent = Array(0..<vcount)
        func find(_ x: Int) -> Int {
            var r = x
            while parent[r] != r { parent[r] = parent[parent[r]]; r = parent[r] }
            var c = x
            while parent[c] != r { let n = parent[c]; parent[c] = r; c = n }
            return r
        }
        func union(_ a: Int, _ b: Int) { let ra = find(a), rb = find(b); if ra != rb { parent[ra] = rb } }

        let idx = mesh.indices
        var t = 0
        while t < idx.count {
            let a = Int(idx[t]), b = Int(idx[t + 1]), c = Int(idx[t + 2])
            union(a, b); union(b, c); t += 3
        }
        var triPerRoot: [Int: Int] = [:]
        t = 0
        while t < idx.count { triPerRoot[find(Int(idx[t])), default: 0] += 1; t += 3 }
        let maxTris = triPerRoot.values.max() ?? 0
        // Keep components that are a real surface, not a speck: at least 0.5% of the
        // largest and an absolute floor of 200 triangles.
        let threshold = max(200, Int(Float(maxTris) * 0.005))
        let keep = Set(triPerRoot.filter { $0.value >= threshold }.map { $0.key })
        guard !keep.isEmpty else { return mesh }

        // Compact kept vertices and remap indices.
        var remap = [Int32](repeating: -1, count: vcount)
        var positions: [SIMD3<Float>] = [], normals: [SIMD3<Float>] = []
        var colors: [SIMD4<UInt8>] = [], indices: [UInt32] = []
        t = 0
        while t < idx.count {
            let a = Int(idx[t]), b = Int(idx[t + 1]), c = Int(idx[t + 2])
            if keep.contains(find(a)) {
                for v in [a, b, c] {
                    if remap[v] < 0 {
                        remap[v] = Int32(positions.count)
                        positions.append(mesh.positions[v]); normals.append(mesh.normals[v])
                        colors.append(mesh.colors[v])
                    }
                    indices.append(UInt32(remap[v]))
                }
            }
            t += 3
        }
        guard !positions.isEmpty else { return mesh }
        return Mesh(positions: positions, normals: normals, colors: colors, indices: indices)
    }

    // MARK: - Rendering

    /// Render one viewport into `color`+`depth`, retrying until MetalSplatter has a
    /// sort ready for this pose (it returns false until then). Returns false if it
    /// never became ready.
    private static func renderOnce(renderer: SplatRenderer, queue: MTLCommandQueue,
                                   viewport: SplatRenderer.ViewportDescriptor,
                                   color: MTLTexture, depth: MTLTexture) throws -> Bool {
        for _ in 0..<40 {
            try Task.checkCancellation()
            guard let cb = queue.makeCommandBuffer() else { return false }
            let didRender = (try? renderer.render(
                viewports: [viewport],
                colorTexture: color, colorStoreAction: .store,
                depthTexture: depth, rasterizationRateMap: nil,
                renderTargetArrayLength: 0, to: cb)) ?? false
            cb.commit()
            cb.waitUntilCompleted()
            if didRender { return true }
            // Sort not ready yet; give the async sort a beat, then retry.
            Thread.sleep(forTimeInterval: 0.02)
        }
        return false
    }

    private static func makeTextures(device: MTLDevice, width: Int, height: Int)
        -> (color: MTLTexture, depth: MTLTexture)? {
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        colorDesc.usage = [.renderTarget, .shaderRead]
        colorDesc.storageMode = .shared
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        depthDesc.usage = [.renderTarget]
        depthDesc.storageMode = .shared
        guard let color = device.makeTexture(descriptor: colorDesc),
              let depth = device.makeTexture(descriptor: depthDesc) else { return nil }
        return (color, depth)
    }

    // MARK: - Geometry helpers

    /// OpenCV (camera +z, +y down) -> glTF (Y-up, -Z forward). Same flip the other
    /// meshers apply, so the fused mesh lands in the identical output space.
    private static func flip(_ v: SIMD3<Float>) -> SIMD3<Float> { SIMD3(v.x, -v.y, -v.z) }

    /// A right-handed projection built directly from pixel focal lengths (so the
    /// renderer's derived focal matches the source view exactly), mapping view-space
    /// z in [-near, -far] to NDC depth [0, 1]. Principal point assumed centred
    /// (cameras.json carries no cx/cy).
    private static func projection(fx: Float, fy: Float, width: Int, height: Int,
                                   near: Float, far: Float) -> simd_float4x4 {
        let zs = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4<Float>(2 * fx / Float(width), 0, 0, 0),
            SIMD4<Float>(0, 2 * fy / Float(height), 0, 0),
            SIMD4<Float>(0, 0, zs, -1),
            SIMD4<Float>(0, 0, zs * near, 0)))
    }

    /// Tight near/far for one camera from the splat positions in front of it, so
    /// 32-bit depth precision concentrates on the actual surface. Falls back to a
    /// scene-scale bracket when too few points are visible.
    /// `clip` is the far cutoff for *fusing* a pixel (median depth * 4, like the
    /// frozen-cloud meshers) so background/floaters don't fuse; near/far are the
    /// projection's depth planes, kept a touch wider so the clip surface still
    /// renders with valid depth.
    private static func depthRange(cameraCenter C: SIMD3<Float>, forward: SIMD3<Float>,
                                   positions: [SIMD3<Float>]) -> (near: Float, far: Float, clip: Float) {
        let step = max(1, positions.count / 20_000)
        var depths: [Float] = []
        depths.reserveCapacity(20_000)
        for k in Swift.stride(from: 0, to: positions.count, by: step) {
            let z = simd_dot(positions[k] - C, forward)   // OpenCV forward = +z
            if z > 1e-4 { depths.append(z) }
        }
        guard depths.count > 8 else {
            // No depth info: bracket by the camera's distance to the point centroid.
            var c = SIMD3<Float>(repeating: 0)
            for p in positions { c += p }
            c /= Float(max(1, positions.count))
            let dist = max(simd_distance(C, c), 1e-3)
            return (dist * 0.05, dist * 4, dist * 4)
        }
        depths.sort()
        let lo = depths[Int(Float(depths.count) * 0.01)]
        let median = depths[depths.count / 2]
        let hi = depths[min(depths.count - 1, Int(Float(depths.count) * 0.99))]
        let near = max(lo * 0.8, hi * 1e-3, 1e-4)
        let far = max(hi * 1.2, near * 1.01)
        let clip = min(far, median * 4)
        return (near, far, clip)
    }

    /// A tight scene box that excludes sparse floaters (which would coarsen the
    /// voxel and waste resolution on empty space). Starts from a generous
    /// percentile box, bins the points into a coarse occupancy grid, then returns
    /// the bounds of the cells dense enough to be real surface (Brush leaves
    /// spread-out low-density floaters that percentile bounds alone don't trim).
    private static func occupancyBounds(_ pts: [SIMD3<Float>]) -> (SIMD3<Float>, SIMD3<Float>) {
        func percentiles(_ vals: [Float], _ lo: Float, _ hi: Float) -> (Float, Float) {
            let s = vals.sorted()
            return (s[Int(Float(s.count) * lo)], s[min(s.count - 1, Int(Float(s.count) * hi))])
        }
        let (xlo, xhi) = percentiles(pts.map { $0.x }, 0.005, 0.995)
        let (ylo, yhi) = percentiles(pts.map { $0.y }, 0.005, 0.995)
        let (zlo, zhi) = percentiles(pts.map { $0.z }, 0.005, 0.995)
        let coarseMin = SIMD3(xlo, ylo, zlo), coarseMax = SIMD3(xhi, yhi, zhi)
        let span = simd_max(coarseMax - coarseMin, SIMD3<Float>(repeating: 1e-6))

        let N = 64
        var counts = [Int](repeating: 0, count: N * N * N)
        func cell(_ p: SIMD3<Float>) -> Int? {
            let f = (p - coarseMin) / span * Float(N)
            let ix = Int(f.x), iy = Int(f.y), iz = Int(f.z)
            guard ix >= 0, ix < N, iy >= 0, iy < N, iz >= 0, iz < N else { return nil }
            return (ix * N + iy) * N + iz
        }
        for p in pts { if let c = cell(p) { counts[c] += 1 } }
        // A cell is "occupied" if it holds a non-trivial share of points; floater
        // cells are far below this. Threshold rides the point count so it scales.
        let threshold = max(2, pts.count / 20_000)
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        let vox = span / Float(N)
        var any = false
        for ix in 0..<N { for iy in 0..<N { for iz in 0..<N {
            guard counts[(ix * N + iy) * N + iz] >= threshold else { continue }
            let lo = coarseMin + SIMD3(Float(ix), Float(iy), Float(iz)) * vox
            mn = simd_min(mn, lo); mx = simd_max(mx, lo + vox); any = true
        }}}
        return any ? (mn, mx) : (coarseMin, coarseMax)
    }

    /// Downscale a camera's image size so its long edge is at most `longEdgeCap`,
    /// preserving aspect. Fusion accuracy comes from combining views, not per-view
    /// resolution, so a moderate cap bounds cost without hurting the surface.
    private static func renderSize(width: Int, height: Int, longEdgeCap: Int) -> (Int, Int) {
        let longEdge = max(width, height)
        guard longEdge > longEdgeCap else { return (width, height) }
        let s = Float(longEdgeCap) / Float(longEdge)
        return (max(1, Int((Float(width) * s).rounded())),
                max(1, Int((Float(height) * s).rounded())))
    }

    /// Pixel stride that keeps a view's deposited surface points near `target`.
    private static func pixelStride(width: Int, height: Int, target: Int) -> Int {
        let ratio = Double(width * height) / Double(max(1, target))
        return max(1, Int(Double(ratio).squareRoot().rounded()))
    }

    /// Evenly sample `keep` items across `items` (endpoints included) so a long
    /// capture fuses a representative spread of angles, not just its first frames.
    private static func evenlySampled<T>(_ items: [T], keep: Int) -> [T] {
        guard items.count > keep, keep > 0 else { return items }
        var out: [T] = []
        out.reserveCapacity(keep)
        for k in 0..<keep {
            out.append(items[Int((Double(k) * Double(items.count - 1) / Double(keep - 1)).rounded())])
        }
        return out
    }
}
