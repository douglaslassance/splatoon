import Foundation
import AppKit

/// Resolves the location of `sharp.mlpackage` (~2.7 GB, lives outside the app).
///
/// Lookup order:
///   1. `SPLATOON_MODEL` environment variable
///   2. A path remembered from a previous pick (UserDefaults)
///   3. `Models/sharp.mlpackage` relative to the current directory
///   4. Prompt the user with an open panel (and remember the choice)
///
/// The app is not sandboxed, so plain file paths are sufficient — no
/// security-scoped bookmarks required.
@MainActor
enum ModelLocator {
    private static let defaultsKey = "sharpModelPath"

    static func resolveModelURL(prompt: Bool = true) -> URL? {
        if let url = rememberedURL() { return url }
        guard prompt else { return nil }
        return promptForModel()
    }

    /// Whether a model can be found without prompting.
    static var hasRememberedModel: Bool { rememberedURL() != nil }

    /// No-op retained for call-site symmetry (was security-scoped access).
    static func endAccess(_ url: URL) {}

    static func forgetModel() {
        UserDefaults.standard.removeObject(forKey: defaultsKey)
    }

    // MARK: - Resolution

    private static func rememberedURL() -> URL? {
        let fm = FileManager.default

        if let envPath = ProcessInfo.processInfo.environment["SPLATOON_MODEL"],
           fm.fileExists(atPath: envPath) {
            return URL(fileURLWithPath: envPath)
        }
        if let savedPath = UserDefaults.standard.string(forKey: defaultsKey),
           fm.fileExists(atPath: savedPath) {
            return URL(fileURLWithPath: savedPath)
        }
        let cwdCandidate = URL(fileURLWithPath: "Models/sharp.mlpackage")
        if fm.fileExists(atPath: cwdCandidate.path) {
            return cwdCandidate.standardizedFileURL
        }
        return nil
    }

    private static func promptForModel() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Locate the SHARP model"
        panel.message = "Select the sharp.mlpackage downloaded via scripts/fetch-model.sh."
        panel.prompt = "Use Model"
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false

        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        UserDefaults.standard.set(url.path, forKey: defaultsKey)
        return url
    }
}
