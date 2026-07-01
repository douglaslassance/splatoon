import Foundation
import AppKit

/// Resolves the location of `sharp.mlpackage`.
///
/// The model is ~2.7 GB and lives outside the app bundle. Because the app is
/// sandboxed, we can't read arbitrary paths — so the first time the user needs
/// it we prompt with an open panel and persist a security-scoped bookmark.
/// Subsequent launches resolve the bookmark without prompting.
@MainActor
enum ModelLocator {
    private static let bookmarkKey = "sharpModelBookmark"

    /// A URL the caller can immediately load. May prompt the user. The returned
    /// URL has already had security-scoped access started; call
    /// `endAccess(_:)` when finished.
    static func resolveModelURL(prompt: Bool = true) -> URL? {
        if let url = resolveBookmark() { return url }
        guard prompt else { return nil }
        return promptForModel()
    }

    /// Whether a remembered model is available without prompting.
    static var hasRememberedModel: Bool {
        UserDefaults.standard.data(forKey: bookmarkKey) != nil
    }

    static func endAccess(_ url: URL) {
        url.stopAccessingSecurityScopedResource()
    }

    // MARK: - Bookmark persistence

    private static func resolveBookmark() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return nil }
        var isStale = false
        guard let url = try? URL(
            resolvingBookmarkData: data,
            options: .withSecurityScope,
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ) else {
            UserDefaults.standard.removeObject(forKey: bookmarkKey)
            return nil
        }
        guard url.startAccessingSecurityScopedResource() else { return nil }
        if isStale { saveBookmark(for: url) }
        return url
    }

    private static func saveBookmark(for url: URL) {
        guard let data = try? url.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        ) else { return }
        UserDefaults.standard.set(data, forKey: bookmarkKey)
    }

    static func forgetModel() {
        UserDefaults.standard.removeObject(forKey: bookmarkKey)
    }

    // MARK: - Prompt

    private static func promptForModel() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Locate the SHARP model"
        panel.message = "Select the sharp.mlpackage downloaded via scripts/fetch-model.sh."
        panel.prompt = "Use Model"
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = []

        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        guard url.startAccessingSecurityScopedResource() else { return nil }
        saveBookmark(for: url)
        return url
    }
}
