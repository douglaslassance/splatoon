import SwiftUI

/// The available mesh-generation strategies, shown in the Settings panel.
enum MeshMethod: String, CaseIterable, Identifiable {
    /// Connect the Gaussian pixel grid into a 2.5D relief surface.
    case grid
    /// One oriented quad ("surfel") per Gaussian, sized and oriented by its shape.
    case surfel
    /// Volumetric reconstruction (TSDF + marching tetrahedra) from the point cloud.
    case poisson

    var id: String { rawValue }

    /// Methods offered for multi-view scene splats. Grid needs SHARP's structured
    /// pixel grid, which scene (COLMAP + OpenSplat) splats don't have, so it's
    /// excluded here.
    static let sceneCases: [MeshMethod] = [.surfel, .poisson]

    var displayName: String {
        switch self {
        case .grid: return "Grid surface"
        case .surfel: return "Surfels (per-splat quads)"
        case .poisson: return "Poisson reconstruction"
        }
    }

    var detail: String {
        switch self {
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
    /// Training steps for the multi-image (COLMAP + OpenSplat) trainer. Too few
    /// and densification never fires (see OpenSplat's warmup-length, 500 steps);
    /// quality keeps improving well past 15000. Higher = longer wait.
    @Published var multiImageIterations: Double {
        didSet { defaults.set(multiImageIterations, forKey: Keys.multiImageIterations) }
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
    /// Poisson: voxel grid resolution along the longest axis.
    @Published var poissonResolution: Double {
        didSet { defaults.set(poissonResolution, forKey: Keys.poissonResolution) }
    }

    /// The mesh method for the given splat kind.
    func method(forScene isScene: Bool) -> MeshMethod {
        isScene ? sceneMethod : singleImageMethod
    }

    /// Changes whenever a meshing setting relevant to `isScene` changes; used to
    /// key the in-app mesh preview cache so it rebuilds when the user tweaks it.
    func signature(forScene isScene: Bool) -> String {
        "\(method(forScene: isScene).rawValue)|\(smoothGrid)|\(depthRatioCull)|\(surfelExtent)|\(poissonResolution)"
    }

    private let defaults = UserDefaults.standard
    private enum Keys {
        static let useMultiImageReconstruction = "reconstruction.useMultiImage"
        static let multiImageIterations = "reconstruction.multiImageIterations"
        static let singleImageMethod = "mesh.singleImageMethod"
        static let sceneMethod = "mesh.sceneMethod"
        static let smoothGrid = "mesh.smoothGrid"
        static let depthRatioCull = "mesh.depthRatioCull"
        static let surfelExtent = "mesh.surfelExtent"
        static let poissonResolution = "mesh.poissonResolution"
    }

    init() {
        useMultiImageReconstruction = defaults.object(forKey: Keys.useMultiImageReconstruction) as? Bool ?? true
        multiImageIterations = defaults.object(forKey: Keys.multiImageIterations) as? Double ?? 15000
        singleImageMethod = MeshMethod(rawValue: defaults.string(forKey: Keys.singleImageMethod) ?? "") ?? .grid
        sceneMethod = MeshMethod(rawValue: defaults.string(forKey: Keys.sceneMethod) ?? "") ?? .poisson
        smoothGrid = defaults.object(forKey: Keys.smoothGrid) as? Bool ?? false
        depthRatioCull = defaults.object(forKey: Keys.depthRatioCull) as? Double ?? 1.5
        surfelExtent = defaults.object(forKey: Keys.surfelExtent) as? Double ?? 2.0
        poissonResolution = defaults.object(forKey: Keys.poissonResolution) as? Double ?? 384
    }
}
