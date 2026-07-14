import SwiftUI

/// The available mesh-generation strategies, shown in the Settings panel.
enum MeshMethod: String, CaseIterable, Identifiable {
    /// Full dense-MVS photogrammetry (COLMAP + OpenMVS) on the registered images:
    /// a watertight, UV-textured mesh. The highest quality, but needs OpenMVS and
    /// the scene's saved images, and takes minutes.
    case photogrammetry
    /// Multi-view TSDF depth fusion: render depth+colour from the trained splat at
    /// each registered camera and fuse the depth maps into a surface.
    case fusion
    /// Anisotropic Gaussian density field -> iso-surface (marching tetrahedra).
    case density
    /// Connect the Gaussian pixel grid into a 2.5D relief surface.
    case grid
    /// One oriented quad ("surfel") per Gaussian, sized and oriented by its shape.
    case surfel
    /// Volumetric reconstruction (TSDF + marching tetrahedra) from the point cloud.
    case poisson

    var id: String { rawValue }

    /// Methods offered for multi-view scene splats. Grid needs SHARP's structured
    /// pixel grid, which scene (COLMAP + OpenSplat) splats don't have, so it's
    /// excluded here. Fusion leads: it renders the splat from the registered
    /// cameras and fuses the depth maps, the same technique the good mesh
    /// pipelines use, so it produces the cleanest surface.
    static let sceneCases: [MeshMethod] = [.photogrammetry, .fusion, .density, .surfel, .poisson]

    /// Methods offered for single-image (SHARP) splats. Photogrammetry and fusion
    /// are excluded: they need multiple registered cameras, which a single photo
    /// doesn't have.
    static let singleImageCases: [MeshMethod] = [.grid, .density, .surfel, .poisson]

    var displayName: String {
        switch self {
        case .photogrammetry: return "Photogrammetry (textured)"
        case .fusion: return "Multi-view fusion"
        case .density: return "Anisotropic density"
        case .grid: return "Grid surface"
        case .surfel: return "Surfels (per-splat quads)"
        case .poisson: return "Poisson reconstruction"
        }
    }

    var detail: String {
        switch self {
        case .photogrammetry:
            return "Runs full dense multi-view stereo (COLMAP + OpenMVS) on the scene's photos to build a "
                + "watertight, UV-textured mesh — by far the best quality. Needs OpenMVS installed and the "
                + "scene's saved images (regenerate older scenes), and takes minutes."
        case .fusion:
            return "Renders depth and colour from the splat at each registered camera, then fuses the "
                + "depth maps (TSDF) into a watertight, vertex-coloured surface. The most accurate on "
                + "multi-image scenes; needs the scene's camera poses."
        case .density:
            return "Builds a density field from each splat's shape and extracts a smooth iso-surface. "
                + "Uses no normals, so it's the most reliable on multi-image scenes."
        case .grid:
            return "Connects the splat's pixel grid into a single 2.5D surface. Fast, on-device."
        case .surfel:
            return "Emits one oriented quad per splat. Faithful to each splat's shape, but not a connected surface."
        case .poisson:
            return "Volumetric reconstruction (TSDF + marching tetrahedra) from the oriented point cloud. "
                + "Smoother and hole-filled. An on-device stand-in for screened Poisson."
        }
    }
}

/// How a single photo becomes a splat. SHARP is a fast on-device CoreML model
/// that produces a 2.5D relief (only the visible surface); TripoSplat is a
/// generative model (the `tripo-cli` tool) that produces a complete 3D object,
/// hallucinating the unseen back and sides — slower, and object-centric.
enum SingleImageGenerator: String, CaseIterable, Identifiable {
    case sharp
    case triposplat

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .sharp:      return "SHARP (fast relief)"
        case .triposplat: return "TripoSplat (full 3D object)"
        }
    }
    var detail: String {
        switch self {
        case .sharp:
            return "Fast on-device model. Produces a 2.5D relief of the photo's visible surface — great for "
                + "scenes and quick results."
        case .triposplat:
            return "Generative model that builds a complete 3D object from one photo, inventing the unseen "
                + "sides. Best for a single object; needs the tripo-cli tool and takes a couple of minutes."
        }
    }
}

/// The Gaussian-splat trainer the multi-image pipeline uses. Both consume the
/// same COLMAP project and produce a standard 3DGS PLY; they differ in backend.
enum SplatTrainer: String, CaseIterable, Identifiable {
    /// libtorch/MPS trainer. The reliable default (ships via fetch-tools.sh), but
    /// its Metal path is crash-prone and it falls back to slow CPU.
    case openSplat
    /// Native wgpu/Metal trainer. No libtorch, no CPU fallback — typically much
    /// faster on Apple GPUs. Optional; built with `cargo` (see fetch-tools.sh).
    case brush

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .openSplat: return "OpenSplat (libtorch)"
        case .brush:     return "Brush (native Metal)"
        }
    }
}

/// The multi-image reconstruction knobs that change the output PLY, bundled so
/// they thread through `open`/`regenerateOpened` together and fold into the
/// splat's cache key as a unit (reopening after any change retrains instead of
/// reusing a stale splat).
struct SceneOptions: Equatable {
    /// Training steps. More = sharper, slower.
    var iterations: Int
    /// Spherical-harmonics degree the trainer uses. Lower means far smaller files
    /// and faster rendering at the cost of some view-dependent shading; degree 1
    /// is plenty for casual captures (degree 3 stores 45 colour coefficients per
    /// splat, degree 1 only 9).
    var shDegree: Int
    /// Solve camera poses with COLMAP's global SfM (`global_mapper`) instead of
    /// its incremental mapper. More robust on sparse, weakly-overlapping captures,
    /// where the incremental mapper often registers only a fraction of the views.
    var globalPoseSolver: Bool
    /// Which trainer runs the splat optimization.
    var trainer: SplatTrainer

    /// Cache-key suffix. Every field changes the resulting PLY, so each belongs
    /// here, matching the old `-i<iters>` key while adding the new knobs.
    var cacheSuffix: String {
        "-i\(iterations)-sh\(shDegree)"
            + (globalPoseSolver ? "-global" : "")
            + (trainer == .brush ? "-brush" : "")
    }
}

/// User-selectable mesh export settings, persisted across launches.
@MainActor
final class MeshSettings: ObservableObject {
    /// When true, opening a photo with several same-place/time siblings
    /// reconstructs them together as a multi-view scene (COLMAP + OpenSplat)
    /// instead of using only the tapped photo. Off falls back to single-image
    /// (SHARP) for every photo.
    @Published var useMultiImageReconstruction: Bool {
        didSet { defaults.set(useMultiImageReconstruction, forKey: Keys.useMultiImageReconstruction) }
    }
    /// How same-place assets are grouped into a scene: same place and time (safe
    /// default), or same place across any day (needs GPS).
    @Published var sceneMatchMode: SceneGrouping.MatchMode {
        didSet { defaults.set(sceneMatchMode.rawValue, forKey: Keys.sceneMatchMode) }
    }
    /// Which generator turns a single photo into a splat: SHARP (fast relief) or
    /// TripoSplat (full 3D object via tripo-cli).
    @Published var singleImageGenerator: SingleImageGenerator {
        didSet { defaults.set(singleImageGenerator.rawValue, forKey: Keys.singleImageGenerator) }
    }
    /// TripoSplat: number of Gaussians to generate (more = more detail, larger,
    /// slightly slower). Up to 262144.
    @Published var triposplatGaussians: Double {
        didSet { defaults.set(triposplatGaussians, forKey: Keys.triposplatGaussians) }
    }
    /// Training steps for the multi-image (COLMAP + OpenSplat) trainer. Too few
    /// and densification never fires (see OpenSplat's warmup-length, 500 steps);
    /// quality keeps improving well past 15000. Higher = longer wait.
    @Published var multiImageIterations: Double {
        didSet { defaults.set(multiImageIterations, forKey: Keys.multiImageIterations) }
    }
    /// Spherical-harmonics degree the multi-view trainer targets. Constrained to
    /// what the viewer's PLY reader accepts: 0 (one flat colour per splat, the
    /// compact default) or 3 (full view-dependent colour, several times larger).
    /// Intermediate degrees are stripped to 0 after training so they still load.
    @Published var sceneSHDegree: Int {
        didSet { defaults.set(sceneSHDegree, forKey: Keys.sceneSHDegree) }
    }
    /// Solve poses with COLMAP's global SfM (`global_mapper`) instead of its
    /// incremental mapper. More robust on sparse, weakly-overlapping captures.
    /// Same COLMAP binary; falls back to the incremental mapper on older builds.
    @Published var useGlobalPoseSolver: Bool {
        didSet { defaults.set(useGlobalPoseSolver, forKey: Keys.useGlobalPoseSolver) }
    }
    /// Which trainer runs the splat optimization. Defaults to OpenSplat (the
    /// always-installed one); Brush is used only if it resolves, else falls back.
    @Published var sceneTrainer: SplatTrainer {
        didSet { defaults.set(sceneTrainer.rawValue, forKey: Keys.sceneTrainer) }
    }
    /// Mesh method for single-image (SHARP) splats — any of the three.
    @Published var singleImageMethod: MeshMethod {
        didSet { defaults.set(singleImageMethod.rawValue, forKey: Keys.singleImageMethod) }
    }
    /// Mesh method for multi-view scene splats — Surfels or Poisson only (Grid
    /// needs a structured pixel grid these splats don't have).
    @Published var sceneMethod: MeshMethod {
        didSet { defaults.set(sceneMethod.rawValue, forKey: Keys.sceneMethod) }
    }
    /// Grid: smooth the surface (Laplacian) to reduce depth noise.
    @Published var smoothGrid: Bool {
        didSet { defaults.set(smoothGrid, forKey: Keys.smoothGrid) }
    }
    /// Grid: occlusion cull aggressiveness (edge depth-jump ratio). Lower = more holes.
    @Published var depthRatioCull: Double {
        didSet { defaults.set(depthRatioCull, forKey: Keys.depthRatioCull) }
    }
    /// Surfel: quad size as a multiple of each splat's radius.
    @Published var surfelExtent: Double {
        didSet { defaults.set(surfelExtent, forKey: Keys.surfelExtent) }
    }
    /// Poisson / Density / Fusion: voxel grid resolution along the longest axis.
    @Published var poissonResolution: Double {
        didSet { defaults.set(poissonResolution, forKey: Keys.poissonResolution) }
    }
    /// Fusion: how many registered cameras to render and fuse. More = fuller
    /// coverage and less noise, but proportionally slower. Views are sampled
    /// evenly across the capture when the scene has more than this.
    @Published var fusionMaxViews: Double {
        didSet { defaults.set(fusionMaxViews, forKey: Keys.fusionMaxViews) }
    }
    /// Density: how tightly the iso-surface hugs the splats (0 = loose/inflated,
    /// 1 = tight to dense cores). Maps to the iso density level.
    @Published var surfaceTightness: Double {
        didSet { defaults.set(surfaceTightness, forKey: Keys.surfaceTightness) }
    }
    /// Density: shifts the surface outward (>0, inflate) or inward (<0).
    @Published var densityOffset: Double {
        didSet { defaults.set(densityOffset, forKey: Keys.densityOffset) }
    }
    /// Photogrammetry: detail level, 0 (fast/coarse) … 1 (full-resolution dense
    /// stereo, slowest). Maps to OpenMVS DensifyPointCloud's resolution level.
    @Published var photogrammetryQuality: Double {
        didSet { defaults.set(photogrammetryQuality, forKey: Keys.photogrammetryQuality) }
    }
    /// Photogrammetry: run OpenMVS's RefineMesh stage for extra geometric detail
    /// (noticeably slower).
    @Published var photogrammetryRefine: Bool {
        didSet { defaults.set(photogrammetryRefine, forKey: Keys.photogrammetryRefine) }
    }

    /// OpenMVS DensifyPointCloud `--resolution-level` from the quality slider:
    /// higher quality → lower level (0 = full image resolution).
    var photogrammetryResolutionLevel: Int {
        photogrammetryQuality > 0.66 ? 0 : (photogrammetryQuality > 0.33 ? 1 : 2)
    }

    /// The current multi-image reconstruction knobs as one value.
    var sceneOptions: SceneOptions {
        SceneOptions(iterations: Int(multiImageIterations),
                     shDegree: sceneSHDegree,
                     globalPoseSolver: useGlobalPoseSolver,
                     trainer: sceneTrainer)
    }

    /// The mesh method for the given splat kind.
    func method(forScene isScene: Bool) -> MeshMethod {
        isScene ? sceneMethod : singleImageMethod
    }

    /// Changes whenever a meshing setting relevant to `isScene` changes; used to
    /// key the in-app mesh preview cache so it rebuilds when the user tweaks it.
    func signature(forScene isScene: Bool) -> String {
        "\(method(forScene: isScene).rawValue)|\(smoothGrid)|\(depthRatioCull)|\(surfelExtent)"
            + "|\(poissonResolution)|\(surfaceTightness)|\(densityOffset)|\(fusionMaxViews)"
            + "|\(photogrammetryQuality)|\(photogrammetryRefine)"
    }

    private let defaults = UserDefaults.standard
    private enum Keys {
        static let useMultiImageReconstruction = "reconstruction.useMultiImage"
        static let sceneMatchMode = "reconstruction.sceneMatchMode"
        static let singleImageGenerator = "generation.singleImageGenerator"
        static let triposplatGaussians = "generation.triposplatGaussians"
        static let multiImageIterations = "reconstruction.multiImageIterations"
        static let sceneSHDegree = "reconstruction.sceneSHDegree"
        static let useGlobalPoseSolver = "reconstruction.useGlobalPoseSolver"
        static let sceneTrainer = "reconstruction.sceneTrainer"
        static let singleImageMethod = "mesh.singleImageMethod"
        static let sceneMethod = "mesh.sceneMethod"
        static let smoothGrid = "mesh.smoothGrid"
        static let depthRatioCull = "mesh.depthRatioCull"
        static let surfelExtent = "mesh.surfelExtent"
        static let poissonResolution = "mesh.poissonResolution"
        static let surfaceTightness = "mesh.surfaceTightness"
        static let densityOffset = "mesh.densityOffset"
        static let fusionMaxViews = "mesh.fusionMaxViews"
        static let photogrammetryQuality = "mesh.photogrammetryQuality"
        static let photogrammetryRefine = "mesh.photogrammetryRefine"
    }

    init() {
        useMultiImageReconstruction = defaults.object(forKey: Keys.useMultiImageReconstruction) as? Bool ?? true
        sceneMatchMode = SceneGrouping.MatchMode(rawValue: defaults.string(forKey: Keys.sceneMatchMode) ?? "")
            ?? .timeAndLocation
        singleImageGenerator = SingleImageGenerator(rawValue: defaults.string(forKey: Keys.singleImageGenerator) ?? "")
            ?? .sharp
        triposplatGaussians = defaults.object(forKey: Keys.triposplatGaussians) as? Double ?? 131072
        multiImageIterations = defaults.object(forKey: Keys.multiImageIterations) as? Double ?? 15000
        sceneSHDegree = (defaults.object(forKey: Keys.sceneSHDegree) as? Int).map { $0 >= 3 ? 3 : 0 } ?? 0
        // Default on: the global SfM solver registers cameras far more reliably than
        // the incremental mapper, whose skewed subset throws off the gravity estimate
        // and tilts the horizon. Falls back to the incremental mapper on older COLMAP.
        useGlobalPoseSolver = defaults.object(forKey: Keys.useGlobalPoseSolver) as? Bool ?? true
        sceneTrainer = SplatTrainer(rawValue: defaults.string(forKey: Keys.sceneTrainer) ?? "") ?? .openSplat
        singleImageMethod = MeshMethod(rawValue: defaults.string(forKey: Keys.singleImageMethod) ?? "") ?? .grid
        sceneMethod = MeshMethod(rawValue: defaults.string(forKey: Keys.sceneMethod) ?? "") ?? .photogrammetry
        smoothGrid = defaults.object(forKey: Keys.smoothGrid) as? Bool ?? false
        depthRatioCull = defaults.object(forKey: Keys.depthRatioCull) as? Double ?? 1.5
        surfelExtent = defaults.object(forKey: Keys.surfelExtent) as? Double ?? 2.0
        poissonResolution = defaults.object(forKey: Keys.poissonResolution) as? Double ?? 384
        surfaceTightness = defaults.object(forKey: Keys.surfaceTightness) as? Double ?? 0.5
        densityOffset = defaults.object(forKey: Keys.densityOffset) as? Double ?? 0.0
        fusionMaxViews = defaults.object(forKey: Keys.fusionMaxViews) as? Double ?? 40
        photogrammetryQuality = defaults.object(forKey: Keys.photogrammetryQuality) as? Double ?? 0.5
        photogrammetryRefine = defaults.object(forKey: Keys.photogrammetryRefine) as? Bool ?? false
    }
}
