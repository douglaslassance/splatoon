import Foundation
import AppKit

/// External command-line tools the multi-image pipeline shells out to.
///
/// COLMAP solves camera poses (sparse reconstruction); OpenSplat trains the
/// Gaussian splat from that COLMAP project. OpenSplat is used over Brush because
/// its CLI is stable and documented (`opensplat <dir> -n <iters> -o out.ply`),
/// and it runs on Apple Metal without CUDA.
enum ReconstructionTool: String, CaseIterable {
    case colmap
    case opensplat

    var displayName: String {
        switch self {
        case .colmap:    return "COLMAP"
        case .opensplat: return "OpenSplat"
        }
    }

    /// Environment override, e.g. `SPLATOON_COLMAP=/path/to/colmap`.
    var envVar: String { "SPLATOON_" + rawValue.uppercased() }
    var defaultsKey: String { "toolPath." + rawValue }
    var installHint: String {
        switch self {
        case .colmap:    return "Install with `brew install colmap`, or run scripts/fetch-tools.sh."
        case .opensplat: return "Build/download OpenSplat via scripts/fetch-tools.sh."
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
            let bundled = resources.appendingPathComponent("bin/\(tool.rawValue)")
            if fm.isExecutableFile(atPath: bundled.path) { return bundled }
        }
        for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
            let candidate = "\(dir)/\(tool.rawValue)"
            if fm.isExecutableFile(atPath: candidate) { return URL(fileURLWithPath: candidate) }
        }
        return nil
    }

    static var isAvailable: Bool {
        ReconstructionTool.allCases.allSatisfy { resolvedURL(for: $0) != nil }
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
