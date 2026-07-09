import Foundation
import Photos
import CoreLocation

/// Decides, at open time, whether a tapped photo is a lone shot (single-image
/// SHARP) or one of several capturing the same place/moment (multi-view
/// reconstruction). Grouping is by burst, then by a timestamp window and, when
/// GPS is present on both, a distance radius.
enum SceneGrouping {

    /// Members within this many seconds of the tapped photo may join its scene.
    static let timeWindow: TimeInterval = 90
    /// When both photos carry GPS, they must be within this many metres.
    static let maxDistance: CLLocationDistance = 25
    /// A group needs at least this many members to be reconstructed as a scene.
    static let minSceneSize = 3

    /// The photos that plausibly capture the same scene as `asset`, including
    /// `asset` itself. `assets` is the full library listing.
    static func neighbors(of asset: PHAsset, in assets: [PHAsset]) -> [PHAsset] {
        let targetDate = asset.creationDate
        let targetLocation = asset.location
        let burst = asset.burstIdentifier

        var group: [PHAsset] = []
        for candidate in assets {
            // Photo grouping is image-only; a video is its own multi-view scene.
            guard candidate.mediaType == .image else { continue }
            if candidate.localIdentifier == asset.localIdentifier {
                group.append(candidate)
                continue
            }
            // A shared burst id is an unambiguous same-scene signal.
            if let burst, candidate.burstIdentifier == burst {
                group.append(candidate)
                continue
            }
            // Otherwise require a close timestamp…
            guard let t0 = targetDate, let t1 = candidate.creationDate,
                  abs(t1.timeIntervalSince(t0)) <= timeWindow else { continue }
            // …and, when GPS is available on both, a close location. Photos without
            // GPS fall back to the time window alone.
            if let l0 = targetLocation, let l1 = candidate.location,
               l1.distance(from: l0) > maxDistance {
                continue
            }
            group.append(candidate)
        }
        return group
    }

    /// Whether a group is large enough to reconstruct as a multi-view scene.
    static func isScene(_ group: [PHAsset]) -> Bool { group.count >= minSceneSize }

    /// A stable cache key for a scene, independent of member order and stable
    /// across launches (a deterministic FNV-1a hash, unlike `String.hashValue`).
    static func sceneKey(for group: [PHAsset]) -> String {
        "scene-" + stableHash(group.map(\.localIdentifier).sorted().joined(separator: "|"))
    }

    /// Deterministic FNV-1a hash of a string, stable across launches. Used for
    /// cache keys (scenes, videos) that must survive relaunches, unlike
    /// `String.hashValue` which is per-process seeded.
    static func stableHash(_ string: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return String(hash, radix: 16)
    }
}
