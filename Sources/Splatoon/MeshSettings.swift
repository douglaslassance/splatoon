import SwiftUI

/// The available mesh-generation strategies, shown in the Settings panel.
enum MeshMethod: String, CaseIterable, Identifiable {
    /// Connect the Gaussian grid into a 2.5D relief surface (default).
    case grid
    /// One oriented quad ("surfel") per Gaussian, sized and oriented by its shape.
    case surfel
    /// Volumetric reconstruction (TSDF + marching tetrahedra) from the point cloud.
    case poisson

    var id: String { rawValue }

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
            return "Connects the splat grid into a single 2.5D surface. Fast, on-device."
        case .surfel:
            return "Emits one oriented quad per splat. Faithful to each splat's shape, but not a connected surface."
        case .poisson:
            return "Volumetric reconstruction (TSDF + marching tetrahedra) from the oriented point cloud. Smoother and hole-filled; on-device stand-in for screened Poisson."
        }
    }
}

/// User-selectable mesh export settings, persisted across launches.
@MainActor
final class MeshSettings: ObservableObject {
    @Published var method: MeshMethod {
        didSet { defaults.set(method.rawValue, forKey: Keys.method) }
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

    private let defaults = UserDefaults.standard
    private enum Keys {
        static let method = "mesh.method"
        static let smoothGrid = "mesh.smoothGrid"
        static let depthRatioCull = "mesh.depthRatioCull"
        static let surfelExtent = "mesh.surfelExtent"
        static let poissonResolution = "mesh.poissonResolution"
    }

    init() {
        let raw = defaults.string(forKey: Keys.method) ?? MeshMethod.grid.rawValue
        method = MeshMethod(rawValue: raw) ?? .grid
        smoothGrid = defaults.object(forKey: Keys.smoothGrid) as? Bool ?? false
        depthRatioCull = defaults.object(forKey: Keys.depthRatioCull) as? Double ?? 1.5
        surfelExtent = defaults.object(forKey: Keys.surfelExtent) as? Double ?? 2.0
        poissonResolution = defaults.object(forKey: Keys.poissonResolution) as? Double ?? 384
    }
}
