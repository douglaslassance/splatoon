import Foundation
import Photos
import CoreLocation

/// Decides, at open time, whether a tapped asset is a lone shot (single-image
/// SHARP) or one of several photos/videos capturing the same place (multi-view
/// reconstruction). Grouping is by burst, then by location and, depending on
/// the match mode, a timestamp window. Photos and videos group together, so a
/// scene can mix stills and clips.
enum SceneGrouping {

    /// How siblings are matched into one scene.
    enum MatchMode: String, CaseIterable, Identifiable {
        /// Same place *and* captured close together (burst / time window / GPS
        /// radius). The safe default: it won't merge unrelated shots that only
        /// share a location.
        case timeAndLocation
        /// Same place only, regardless of when. Groups shots of one place taken
        /// on different days. Needs GPS on both assets; without it there's no way
        /// to tell same-place captures apart across time.
        case locationOnly

        var id: String { rawValue }
        var displayName: String {
            switch self {
            case .timeAndLocation: return "Time and location"
            case .locationOnly: return "Location only"
            }
        }
    }

    /// Members within this many seconds of the tapped asset may join its scene.
    static let timeWindow: TimeInterval = 90
    /// When both assets carry GPS, they must be within this many metres.
    static let maxDistance: CLLocationDistance = 25
    /// A photo-only group needs at least this many members to be a scene. (A
    /// video is a scene on its own, since it yields many frames.)
    static let minSceneSize = 3

    /// The photos and videos that plausibly capture the same scene as `asset`,
    /// including `asset` itself. `assets` is the full library listing.
    static func neighbors(of asset: PHAsset, in assets: [PHAsset],
                          matchMode: MatchMode = .timeAndLocation) -> [PHAsset] {
        let targetDate = asset.creationDate
        let targetLocation = asset.location
        let burst = asset.burstIdentifier

        var group: [PHAsset] = []
        for candidate in assets {
            if candidate.localIdentifier == asset.localIdentifier {
                group.append(candidate)
                continue
            }
            // A shared burst id is an unambiguous same-scene signal.
            if let burst, candidate.burstIdentifier == burst {
                group.append(candidate)
                continue
            }
            switch matchMode {
            case .timeAndLocation:
                // Require a close timestamp…
                guard let t0 = targetDate, let t1 = candidate.creationDate,
                      abs(t1.timeIntervalSince(t0)) <= timeWindow else { continue }
                // …and, when GPS is available on both, a close location. Assets
                // without GPS fall back to the time window alone.
                if let l0 = targetLocation, let l1 = candidate.location,
                   l1.distance(from: l0) > maxDistance {
                    continue
                }
                group.append(candidate)
            case .locationOnly:
                // Same place regardless of time. Both must carry GPS.
                guard let l0 = targetLocation, let l1 = candidate.location,
                      l1.distance(from: l0) <= maxDistance else { continue }
                group.append(candidate)
            }
        }
        return group
    }

    /// Whether a group warrants multi-view reconstruction: any video yields many
    /// overlapping frames on its own; otherwise it takes at least `minSceneSize`
    /// photos.
    static func isScene(_ group: [PHAsset]) -> Bool {
        group.contains { $0.mediaType == .video } || group.count >= minSceneSize
    }

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
