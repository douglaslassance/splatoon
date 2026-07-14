import Foundation

extension Notification.Name {
    /// Posted after the on-disk splat cache is emptied, so the gallery can drop
    /// its "has splat" badges without waiting for a relaunch.
    static let splatCacheCleared = Notification.Name("SplatoonSplatCacheCleared")
}

/// The on-disk cache of generated splats (and their cameras.json / mesh
/// sidecars). Single source of truth for where it lives, how big it is, and how
/// to empty it, shared by `GalleryModel` (writer) and the Settings Cache tab.
enum SplatCache {
    static var directory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Splatoon/Splats", isDirectory: true)
    }

    /// Total bytes of every file in the cache directory.
    static func size() -> Int64 {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: directory,
                                                      includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }
        return items.reduce(0) { sum, url in
            sum + Int64((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
    }

    /// Delete every cached file (leaving the directory), then broadcast so the
    /// gallery refreshes. Best-effort per file.
    static func clear() {
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        for url in items { try? fm.removeItem(at: url) }
        NotificationCenter.default.post(name: .splatCacheCleared, object: nil)
    }
}
