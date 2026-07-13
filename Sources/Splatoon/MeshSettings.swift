import SwiftUI

/// The available mesh-generation strategies, shown in the Settings panel.
enum MeshMethod: String, CaseIterable, Identifiable {
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
    /// excluded here. Density leads: it needs no normals, so it holds up best on
    /// unstructured multi-view splats.
    static let sceneCases: [MeshMethod] = [.density, .surfel, .poisson]

    var displayName: String {
        switch self {
        case .density: return "Anisotropic density"
        case .grid: return "Grid surface"
        case .surfel: return "Surfels (per-splat quads)"
        case .poisson: return "Poisson reconstruction"
        }
    }

    var detail: String {
        switch self {
        case .density:
            return "Builds a density field from each splat's shape and extracts a smooth iso-surface. "
                + "Uses no normals, so it's the most reliable on multi-image scenes."
        case .grid:
            return "Connects the splat's pixel grid into a single 2.5D surface. Fast, on-device."
        case .surfel:
            return "Emits one oriented quad per splat. Faithful to each splat's shape, but not a connected surface."
        case .poisson:
            return "Volumetric reconstruction (TSDF + marching tetrahedra) from the oriented point cloud. "
                + "Smoother and hole-filled; an on-device stand-in for screened Poisson."
        }
    }
}

/// The multi-image reconstruction knobs that change the output PLY, bundled so
/// they thread through `open`/`regenerateOpened` together and fold into the
/// splat's cache key as a unit (reopening after any change retrains instead of
/// reusing a stale splat).
struct SceneOptions: Equatable {
    /// OpenSplat training steps. More = sharper, slower.
    var iterations: Int
    /// Spherical-harmonics degree OpenSplat trains, 1…3. Lower means far smaller
    /// files and faster rendering at the cost of some view-dependent shading;
    /// degree 1 is plenty for casual captures (degree 3 stores 45 colour
    /// coefficients per splat, degree 1 only 9).
    var shDegree: Int
    /// Solve camera poses with COLMAP's global SfM (`global_mapper`) instead of
    /// its incremental mapper. More robust on sparse, weakly-overlapping captures,
    /// where the incremental mapper often registers only a fraction of the views.
    var globalPoseSolver: Bool

    /// Cache-key suffix. Every field changes the resulting PLY, so each belongs
    /// here, matching the old `-i<iters>` key while adding the new knobs.
    var cacheSuffix: String {
        "-i\(iterations)-sh\(shDegree)" + (globalPoseSolver ? "-global" : "")
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
    /// Training steps for the multi-image (COLMAP + OpenSplat) trainer. Too few
    /// and densification never fires (see OpenSplat's warmup-length, 500 steps);
    /// quality keeps improving well past 15000. Higher = longer wait.
    @Published var multiImageIterations: Double {
        didSet { defaults.set(multiImageIterations, forKey: Keys.multiImageIterations) }
    }
    /// Spherical-harmonics degree the multi-view trainer uses (1…3). Defaults to
    /// 1: casual captures gain little from higher-order view-dependent colour, and
    /// each extra degree multiplies the per-splat colour storage (degree 3 splats
    /// are several times larger on disk and slower to render).
    @Published var sceneSHDegree: Int {
        didSet { defaults.set(sceneSHDegree, forKey: Keys.sceneSHDegree) }
    }
    /// Solve poses with COLMAP's global SfM (`global_mapper`) instead of its
    /// incremental mapper. More robust on sparse, weakly-overlapping captures.
    /// Same COLMAP binary; falls back to the incremental mapper on older builds.
    @Published var useGlobalPoseSolver: Bool {
        didSet { defaults.set(useGlobalPoseSolver, forKey: Keys.useGlobalPoseSolver) }
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
    /// Poisson / Density: voxel grid resolution along the longest axis.
    @Published var poissonResolution: Double {
        didSet { defaults.set(poissonResolution, forKey: Keys.poissonResolution) }
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

    /// The current multi-image reconstruction knobs as one value.
    var sceneOptions: SceneOptions {
        SceneOptions(iterations: Int(multiImageIterations),
                     shDegree: sceneSHDegree,
                     globalPoseSolver: useGlobalPoseSolver)
    }

    /// The mesh method for the given splat kind.
    func method(forScene isScene: Bool) -> MeshMethod {
        isScene ? sceneMethod : singleImageMethod
    }

    /// Changes whenever a meshing setting relevant to `isScene` changes; used to
    /// key the in-app mesh preview cache so it rebuilds when the user tweaks it.
    func signature(forScene isScene: Bool) -> String {
        "\(method(forScene: isScene).rawValue)|\(smoothGrid)|\(depthRatioCull)|\(surfelExtent)"
            + "|\(poissonResolution)|\(surfaceTightness)|\(densityOffset)"
    }

    private let defaults = UserDefaults.standard
    private enum Keys {
        static let useMultiImageReconstruction = "reconstruction.useMultiImage"
        static let sceneMatchMode = "reconstruction.sceneMatchMode"
        static let multiImageIterations = "reconstruction.multiImageIterations"
        static let sceneSHDegree = "reconstruction.sceneSHDegree"
        static let useGlobalPoseSolver = "reconstruction.useGlobalPoseSolver"
        static let singleImageMethod = "mesh.singleImageMethod"
        static let sceneMethod = "mesh.sceneMethod"
        static let smoothGrid = "mesh.smoothGrid"
        static let depthRatioCull = "mesh.depthRatioCull"
        static let surfelExtent = "mesh.surfelExtent"
        static let poissonResolution = "mesh.poissonResolution"
        static let surfaceTightness = "mesh.surfaceTightness"
        static let densityOffset = "mesh.densityOffset"
    }

    init() {
        useMultiImageReconstruction = defaults.object(forKey: Keys.useMultiImageReconstruction) as? Bool ?? true
        sceneMatchMode = SceneGrouping.MatchMode(rawValue: defaults.string(forKey: Keys.sceneMatchMode) ?? "")
            ?? .timeAndLocation
        multiImageIterations = defaults.object(forKey: Keys.multiImageIterations) as? Double ?? 15000
        sceneSHDegree = (defaults.object(forKey: Keys.sceneSHDegree) as? Int).map { min(3, max(1, $0)) } ?? 1
        useGlobalPoseSolver = defaults.object(forKey: Keys.useGlobalPoseSolver) as? Bool ?? false
        singleImageMethod = MeshMethod(rawValue: defaults.string(forKey: Keys.singleImageMethod) ?? "") ?? .grid
        sceneMethod = MeshMethod(rawValue: defaults.string(forKey: Keys.sceneMethod) ?? "") ?? .density
        smoothGrid = defaults.object(forKey: Keys.smoothGrid) as? Bool ?? false
        depthRatioCull = defaults.object(forKey: Keys.depthRatioCull) as? Double ?? 1.5
        surfelExtent = defaults.object(forKey: Keys.surfelExtent) as? Double ?? 2.0
        poissonResolution = defaults.object(forKey: Keys.poissonResolution) as? Double ?? 384
        surfaceTightness = defaults.object(forKey: Keys.surfaceTightness) as? Double ?? 0.5
        densityOffset = defaults.object(forKey: Keys.densityOffset) as? Double ?? 0.0
    }
}
