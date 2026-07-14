import Foundation
import AppKit

/// External command-line tools the multi-image pipeline shells out to.
///
/// COLMAP solves camera poses (sparse reconstruction); a trainer turns that
/// COLMAP project into a Gaussian splat. OpenSplat (libtorch/MPS) is the default,
/// always-installed trainer; Brush (native wgpu/Metal, headless `brush-cli`) is
/// an opt-in alternative selected by the "Trainer" setting.
enum ReconstructionTool: String, CaseIterable {
    case colmap
    case opensplat
    /// Optional native-Metal trainer, selected by the "Trainer" setting. Not in
    /// `required`, so its absence doesn't gate the multi-image feature.
    case brush
    // OpenMVS binaries for the photogrammetry mesh pipeline (opt-in; each ships as
    // its own executable). Not in `required` — their absence only disables the
    // photogrammetry mesh method, not scene reconstruction.
    case interfaceCOLMAP
    case densifyPointCloud
    case reconstructMesh
    case refineMesh
    case textureMesh

    /// The tools the pipeline always needs (COLMAP + the default trainer). Brush
    /// and the OpenMVS tools are opt-in, so they're excluded.
    static let required: [ReconstructionTool] = [.colmap, .opensplat]

    /// The OpenMVS executables the photogrammetry pipeline *requires* (COLMAP
    /// supplies `image_undistorter` separately). `refineMesh` is excluded: it's an
    /// optional stage, only invoked when the user enables Refine, so its absence
    /// must not disable the whole feature.
    static let openMVS: [ReconstructionTool] =
        [.interfaceCOLMAP, .densifyPointCloud, .reconstructMesh, .textureMesh]

    var displayName: String {
        switch self {
        case .colmap:            return "COLMAP"
        case .opensplat:         return "OpenSplat"
        case .brush:             return "Brush"
        case .interfaceCOLMAP:   return "OpenMVS InterfaceCOLMAP"
        case .densifyPointCloud: return "OpenMVS DensifyPointCloud"
        case .reconstructMesh:   return "OpenMVS ReconstructMesh"
        case .refineMesh:        return "OpenMVS RefineMesh"
        case .textureMesh:       return "OpenMVS TextureMesh"
        }
    }

    /// The executable's filename on disk (differs from the case name for Brush,
    /// whose headless binary is `brush-cli`; the OpenMVS tools keep their
    /// CamelCase executable names).
    var binaryName: String {
        switch self {
        case .colmap:            return "colmap"
        case .opensplat:         return "opensplat"
        case .brush:             return "brush-cli"
        case .interfaceCOLMAP:   return "InterfaceCOLMAP"
        case .densifyPointCloud: return "DensifyPointCloud"
        case .reconstructMesh:   return "ReconstructMesh"
        case .refineMesh:        return "RefineMesh"
        case .textureMesh:       return "TextureMesh"
        }
    }

    /// Environment override, e.g. `SPLATOON_COLMAP=/path/to/colmap`.
    var envVar: String { "SPLATOON_" + rawValue.uppercased() }
    var defaultsKey: String { "toolPath." + rawValue }
    var installHint: String {
        switch self {
        case .colmap:    return "Install with `brew install colmap`, or run scripts/fetch-tools.sh."
        case .opensplat: return "Build/download OpenSplat via scripts/fetch-tools.sh."
        case .brush:     return "Install with `cargo install --git https://github.com/ArthurBrussee/brush brush-cli`, or run scripts/fetch-tools.sh."
        case .interfaceCOLMAP, .densifyPointCloud, .reconstructMesh, .refineMesh, .textureMesh:
            return "Build OpenMVS via scripts/fetch-tools.sh (for the photogrammetry mesh)."
        }
    }
}

/// Resolves the reconstruction tool binaries, mirroring `ModelLocator`: the heavy
/// tools live outside the app and are found at runtime rather than bundled.
///
/// Lookup order: `SPLATOON_<TOOL>` env var → remembered pick (UserDefaults) →
/// bundled `Contents/Resources/bin/<tool>` → common install dirs → prompt.
enum ToolLocator {

    /// Resolve without any UI. Safe to call off the main actor (e.g. from the
    /// reconstruction task). Returns nil if the tool isn't found anywhere.
    static func resolvedURL(for tool: ReconstructionTool) -> URL? {
        let fm = FileManager.default

        if let path = ProcessInfo.processInfo.environment[tool.envVar],
           fm.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        if let path = UserDefaults.standard.string(forKey: tool.defaultsKey),
           fm.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        if let resources = Bundle.main.resourceURL {
            let bundled = resources.appendingPathComponent("bin/\(tool.binaryName)")
            if fm.isExecutableFile(atPath: bundled.path) { return bundled }
        }
        // ~/.cargo/bin is where `cargo install` drops Brush; the rest are the
        // usual Homebrew/system locations.
        let cargoBin = fm.homeDirectoryForCurrentUser.appendingPathComponent(".cargo/bin").path
        for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", cargoBin] {
            let candidate = "\(dir)/\(tool.binaryName)"
            if fm.isExecutableFile(atPath: candidate) { return URL(fileURLWithPath: candidate) }
        }
        return nil
    }

    static var isAvailable: Bool {
        ReconstructionTool.required.allSatisfy { resolvedURL(for: $0) != nil }
    }

    /// Whether the OpenMVS photogrammetry pipeline can run: COLMAP (for
    /// `image_undistorter`) plus every OpenMVS binary must resolve.
    static var photogrammetryAvailable: Bool {
        resolvedURL(for: .colmap) != nil
            && ReconstructionTool.openMVS.allSatisfy { resolvedURL(for: $0) != nil }
    }

    /// Resolve, prompting the user to locate the binary if it isn't found. Must
    /// run on the main actor (shows an open panel). Remembers the chosen path.
    @MainActor
    static func resolveOrPrompt(for tool: ReconstructionTool) -> URL? {
        if let url = resolvedURL(for: tool) { return url }

        let panel = NSOpenPanel()
        panel.title = "Locate \(tool.displayName)"
        panel.message = "Select the \(tool.rawValue) executable. \(tool.installHint)"
        panel.prompt = "Use \(tool.displayName)"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.showsHiddenFiles = true

        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        UserDefaults.standard.set(url.path, forKey: tool.defaultsKey)
        return url
    }
}
