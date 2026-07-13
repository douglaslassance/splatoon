import SwiftUI
import AppKit
import Photos
import ImageIO
import CoreImage
import CoreGraphics
import AVFoundation
import UniformTypeIdentifiers

/// One shared, reused CIContext. Creating a CIContext per image (as this code
/// used to) is a documented anti-pattern — each allocates Metal resources and an
/// internal cache — and over a session of 24 MP decodes it leaks heavily.
/// CIContext is thread-safe, so a single shared instance is correct.
private let sharedCIContext = CIContext()

/// Drives the Photos-library gallery: authorization, asset listing, thumbnail
/// image manager, per-image splat generation, and an on-disk splat cache.
@MainActor
final class GalleryModel: ObservableObject {

    struct OpenedSplat: Identifiable, Equatable {
        let id: String        // PHAsset.localIdentifier, or a scene key
        let title: String
        let url: URL?         // nil while the splat is still being generated
        var isScene = false   // reconstructed from multiple photos (unstructured)
        /// Where the fly camera should start. nil for SHARP splats (world origin
        /// already is the photo's own viewpoint); a real registered camera pose
        /// for scenes (COLMAP's world origin is otherwise arbitrary).
        var initialPose: ScenePose?
    }

    /// Live status of an in-progress splat generation — single-image (SHARP) or
    /// multi-image (COLMAP + OpenSplat) — surfaced in the docked progress bar so
    /// both share the same UI. Independent of `opened`/navigation, so a
    /// long-running scene keeps updating even after the user backs to the gallery.
    struct SceneProgress: Equatable {
        var key: String
        var title: String
        var stageLabel: String
        var fraction: Double     // 0...1 across the whole pipeline
        var isComplete = false
        var isScene = true       // scenes reopen at a registered camera pose; SHARP doesn't
        var cancellable = true   // multi-image can be cancelled; SHARP is too fast to bother
        var indeterminate = false // no measurable fraction (splat load, mesh build)
        var elapsed: String?     // human compute time, shown once complete
    }

    @Published private(set) var authorization: PHAuthorizationStatus =
        PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @Published private(set) var assets: [PHAsset] = []
    @Published private(set) var splatIdentifiers: Set<String> = []
    /// A scene reconstruction that failed but can still fall back to single-image
    /// SHARP on the asset the user tapped. Drives a recoverable alert (Use Single
    /// Image / Cancel) instead of a dead-end "OK".
    struct SceneFailure: Identifiable {
        let message: String
        let asset: PHAsset
        var id: String { asset.localIdentifier }
    }

    @Published var opened: OpenedSplat?
    @Published private(set) var sceneProgress: SceneProgress?
    @Published var sceneFailure: SceneFailure?
    @Published var errorMessage: String?

    /// The mesh preview for the currently opened splat, built lazily when the
    /// detail view switches to Mesh mode. `openedMeshKey` identifies which
    /// (splat, mesh-settings) combination it was built for.
    @Published private(set) var openedMesh: Mesh?
    private var openedMeshKey: String?

    /// Fraction of Gaussians to keep (1.0 = all).
    var decimation: Float = 1.0

    /// The source asset (tapped photo / video) each opened splat was built from,
    /// keyed by the splat's cache id — so it can be regenerated on demand.
    private var sourceForOpened: [String: PHAsset] = [:]

    /// Wall-clock start of an in-flight build (SHARP generation or multi-image
    /// reconstruction), keyed like `sceneProgress`, to report compute time on
    /// completion.
    private var buildStartedAt: [String: Date] = [:]

    /// Compute time as a compact human string ("2.4s", "3m 12s").
    private static func elapsedString(since start: Date) -> String {
        let seconds = Date().timeIntervalSince(start)
        if seconds < 60 { return String(format: "%.1fs", seconds) }
        let total = Int(seconds.rounded())
        return "\(total / 60)m \(total % 60)s"
    }

    let imageManager = PHCachingImageManager()

    private let cacheDir: URL
    private var cachedRunner: SharpModelRunner?
    private var cachedRunnerURL: URL?
    private var sceneReconstructor: MultiImageReconstructor?

    init() {
        cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Splatoon/Splats", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        loadCacheIndex()
    }

    // MARK: - Authorization & fetch

    func onAppear() {
        if authorization == .authorized || authorization == .limited {
            if assets.isEmpty { fetchAssets() }
        }
    }

    func requestAccess() {
        PHPhotoLibrary.requestAuthorization(for: .readWrite) { status in
            Task { @MainActor in
                self.authorization = status
                if status == .authorized || status == .limited {
                    self.fetchAssets()
                }
            }
        }
    }

    private func fetchAssets() {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.predicate = NSPredicate(format: "mediaType == %d OR mediaType == %d",
                                        PHAssetMediaType.image.rawValue, PHAssetMediaType.video.rawValue)
        let result = PHAsset.fetchAssets(with: options)
        var collected: [PHAsset] = []
        collected.reserveCapacity(result.count)
        result.enumerateObjects { asset, _, _ in collected.append(asset) }
        assets = collected
    }

    // MARK: - Cache

    func hasSplat(_ asset: PHAsset) -> Bool {
        splatIdentifiers.contains(asset.localIdentifier)
    }

    private func splatURL(for identifier: String) -> URL {
        // localIdentifiers contain "/" (never "~"), so "~" is a safe, reversible
        // substitute for a flat filename.
        cacheDir.appendingPathComponent(identifier.replacingOccurrences(of: "/", with: "~") + ".ply")
    }

    private func loadCacheIndex() {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: cacheDir.path)) ?? []
        splatIdentifiers = Set(
            files.filter { $0.hasSuffix(".ply") }
                 .map { String($0.dropLast(4)).replacingOccurrences(of: "~", with: "/") }
        )
    }

    // MARK: - Open / generate

    /// Open an asset. Same-place photos and videos (per `matchMode`) reconstruct
    /// together as a multi-view scene: a video yields many overlapping frames on
    /// its own, and stills and clips can mix in one scene. A lone photo with no
    /// siblings goes through single-image SHARP. Scenes are gated by
    /// `allowMultiImage` (the "Reconstruct multi-input scenes" setting).
    func open(_ asset: PHAsset, allowMultiImage: Bool = true,
              options: SceneOptions = SceneOptions(iterations: 15000, shDegree: 1,
                                                   globalPoseSolver: false, trainer: .openSplat),
              matchMode: SceneGrouping.MatchMode = .timeAndLocation) {
        errorMessage = nil
        resetOpenedMesh()

        if allowMultiImage {
            let group = SceneGrouping.neighbors(of: asset, in: assets, matchMode: matchMode)
            if SceneGrouping.isScene(group) {
                openScene(group, hero: asset, options: options)
                return
            }
        }

        // Only a lone photo falls through to single-image SHARP. A video that
        // isn't a scene means multi-input is off, so it has no single-image path
        // here (its SHARP fallback runs on the sharpest frame, on demand).
        if asset.mediaType == .video {
            errorMessage = "Turn on “Reconstruct multi-input scenes” in Settings (⌘,) to build a splat from a video."
            return
        }

        let identifier = asset.localIdentifier
        let destination = splatURL(for: identifier)
        let title = Self.title(for: asset)
        sourceForOpened[identifier] = asset

        if FileManager.default.fileExists(atPath: destination.path) {
            opened = OpenedSplat(id: identifier, title: title, url: destination)
            return
        }

        guard let modelURL = ModelLocator.resolveModelURL() else {
            errorMessage = "No SHARP model found. Run scripts/fetch-model.sh, then choose sharp.mlpackage."
            return
        }

        // Switch to the splat view immediately (url nil = still generating);
        // progress shows in the docked bar, same as multi-image scenes.
        opened = OpenedSplat(id: identifier, title: title, url: nil)
        sceneProgress = SceneProgress(key: identifier, title: title, stageLabel: "Loading photo…",
                                      fraction: 0.05, isScene: false, cancellable: false)
        buildStartedAt[identifier] = Date()
        let requestOptions = PHImageRequestOptions()
        requestOptions.isNetworkAccessAllowed = true
        requestOptions.deliveryMode = .highQualityFormat
        requestOptions.isSynchronous = false

        imageManager.requestImageDataAndOrientation(for: asset, options: requestOptions) { [weak self] data, _, _, _ in
            Task { @MainActor in
                guard let self else { return }
                guard let data, let cgImage = Self.decodeOriented(data) else {
                    self.failGeneration(key: identifier, message: "Could not load that photo.")
                    return
                }
                self.generate(cgImage: cgImage,
                              identifier: identifier,
                              title: title,
                              modelURL: modelURL,
                              destination: destination)
            }
        }
    }

    private func generate(cgImage: CGImage,
                          identifier: String,
                          title: String,
                          modelURL: URL,
                          destination: URL) {
        // SHARP has no fine-grained progress (Core ML inference is one opaque
        // call), so the fractions are coarse stage markers — but the shared bar
        // still gives named stages instead of a bare spinner.
        updateGenerationStage(key: identifier, "Loading model…", 0.15)
        let decimation = self.decimation
        let existingRunner = (cachedRunnerURL == modelURL) ? cachedRunner : nil

        Task.detached(priority: .userInitiated) {
            do {
                let runner = try existingRunner ?? SharpModelRunner(modelURL: modelURL)
                await MainActor.run {
                    self.cachedRunner = runner
                    self.cachedRunnerURL = modelURL
                    self.updateGenerationStage(key: identifier, "Preparing image…", 0.35)
                }
                let input = try runner.preprocess(cgImage)

                await MainActor.run { self.updateGenerationStage(key: identifier, "Running inference…", 0.6) }
                let gaussians = try runner.predict(image: input, focalLengthPx: 1536)

                await MainActor.run { self.updateGenerationStage(key: identifier, "Writing splat…", 0.9) }
                try SplatExporter.savePLY(
                    gaussians: gaussians,
                    focalLengthPx: 1536,
                    imageShape: (height: runner.inputHeight, width: runner.inputWidth),
                    to: destination,
                    decimation: decimation
                )

                await MainActor.run {
                    self.splatIdentifiers.insert(identifier)
                    // Only reveal the splat if the user is still on this one
                    // (they may have navigated back during generation).
                    if self.opened?.id == identifier {
                        self.opened = OpenedSplat(id: identifier, title: title, url: destination)
                    }
                    self.completeGeneration(key: identifier, title: title, isScene: false)
                }
            } catch {
                await MainActor.run {
                    self.failGeneration(key: identifier, message: "Generation failed: \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: - Generation progress (shared by single- and multi-image)

    private func updateGenerationStage(key: String, _ stage: String, _ fraction: Double) {
        guard sceneProgress?.key == key else { return }
        sceneProgress?.stageLabel = stage
        sceneProgress?.fraction = fraction
    }

    /// Finish a generation: reveal a brief "Ready" state in the docked bar (which
    /// auto-dismisses, and lets a user who wandered off tap back in).
    private func completeGeneration(key: String, title: String, isScene: Bool) {
        guard sceneProgress?.key == key else { return }
        let elapsed = buildStartedAt.removeValue(forKey: key).map(Self.elapsedString(since:))
        sceneProgress = SceneProgress(key: key, title: title, stageLabel: "Ready", fraction: 1,
                                      isComplete: true, isScene: isScene, cancellable: false, elapsed: elapsed)
    }

    private func failGeneration(key: String, message: String) {
        if sceneProgress?.key == key { sceneProgress = nil }
        errorMessage = message
        if opened?.id == key { opened = nil }
    }

    /// Navigate back to the gallery. Any in-flight scene reconstruction keeps
    /// running in the background — `sceneProgress` keeps updating regardless —
    /// so a long training run isn't lost just by looking at something else.
    func closeOpened() {
        opened = nil
        resetOpenedMesh()
        clearIndeterminate()   // drop a stray "Loading splat…"/"Building mesh…" bar
    }

    /// Discard the opened splat's cached files and rebuild it from the original
    /// source (photo, photo group, or video). Uses the current settings.
    func regenerateOpened(options: SceneOptions, matchMode: SceneGrouping.MatchMode = .timeAndLocation) {
        guard let opened, let asset = sourceForOpened[opened.id] else { return }
        let wasScene = opened.isScene

        let ply = splatURL(for: opened.id)
        try? FileManager.default.removeItem(at: ply)
        try? FileManager.default.removeItem(at: MultiImageReconstructor.camerasURL(for: ply))
        splatIdentifiers.remove(opened.id)
        sourceForOpened[opened.id] = nil
        resetOpenedMesh()
        self.opened = nil

        if asset.mediaType == .video && !wasScene {
            openVideoSingleFrame(asset)                                   // the video's SHARP fallback
        } else {
            open(asset, allowMultiImage: wasScene, options: options, matchMode: matchMode)
        }
    }

    /// Stop the in-flight reconstruction, if any (the docked progress bar's
    /// Cancel button). Unlike navigating away, this really does discard it.
    func cancelSceneReconstruction() {
        sceneReconstructor?.cancel()
    }

    /// Bring the scene tracked by `sceneProgress` back on screen — either its
    /// still-generating placeholder (drives the same in-context progress view)
    /// or, once cached, the finished splat.
    func reopenInProgressScene() {
        guard let progress = sceneProgress else { return }
        let destination = splatURL(for: progress.key)
        if FileManager.default.fileExists(atPath: destination.path) {
            opened = OpenedSplat(id: progress.key, title: progress.title, url: destination, isScene: progress.isScene,
                                 initialPose: progress.isScene ? MultiImageReconstructor.initialPose(for: destination) : nil)
        } else {
            opened = OpenedSplat(id: progress.key, title: progress.title, url: nil, isScene: progress.isScene)
        }
    }

    /// Dismiss the "Ready" state of a finished reconstruction without opening it.
    func dismissSceneProgress() {
        sceneProgress = nil
    }

    // MARK: - Multi-image scene reconstruction

    /// Reconstruct a splat from one or more same-place sources (photos and/or
    /// videos), caching the result like a single-image splat. Photos contribute
    /// one frame each; videos are frame-sampled. All frames are quality-scored and
    /// culled before COLMAP sees them (see `selectAndWriteFrames`).
    private func openScene(_ sources: [PHAsset], hero: PHAsset, options: SceneOptions) {
        // The reconstruction knobs are part of the cache key so changing any of
        // them in Settings and reopening actually retrains, instead of silently
        // reusing a splat built with the old settings.
        let key = SceneGrouping.sceneKey(for: sources) + options.cacheSuffix
        sourceForOpened[key] = hero
        // A shared camera intrinsic is a strong prior when every frame comes from
        // one capture (a single video, or a photo group from one phone). Mixed or
        // multi-clip scenes span cameras, so let COLMAP solve intrinsics per image.
        let videoCount = sources.filter { $0.mediaType == .video }.count
        let sharedCamera = videoCount == 0 || (videoCount == 1 && sources.count == 1)
        startSceneReconstruction(key: key, title: Self.sceneTitle(for: sources),
                                 options: options, hero: hero, sharedCamera: sharedCamera) { imagesDir, report in
            await self.selectAndWriteFrames(from: sources, to: imagesDir, report: report)
        }
    }

    /// A human title for a scene built from `sources`.
    private static func sceneTitle(for sources: [PHAsset]) -> String {
        let videos = sources.filter { $0.mediaType == .video }.count
        let photos = sources.count - videos
        if videos == 0 { return "Scene · \(photos) photos" }
        if photos == 0 { return videos == 1 ? "Video scene" : "Video scene · \(videos) clips" }
        return "Scene · \(photos) photo\(photos == 1 ? "" : "s") + \(videos) video\(videos == 1 ? "" : "s")"
    }

    /// Recover from a failed scene by running single-image SHARP on the asset the
    /// user tapped (its sharpest frame, for a video). Driven by the failure alert.
    func fallbackToSingleImage() {
        guard let failure = sceneFailure else { return }
        let asset = failure.asset
        sceneFailure = nil
        if asset.mediaType == .video {
            openVideoSingleFrame(asset)
        } else {
            open(asset, allowMultiImage: false)   // straight to SHARP, no re-grouping
        }
    }

    /// SHARP on a video's sharpest frame, the single-image fallback for a video.
    private func openVideoSingleFrame(_ asset: PHAsset) {
        resetOpenedMesh()
        let identifier = asset.localIdentifier
        let destination = splatURL(for: identifier)
        let title = Self.title(for: asset)
        sourceForOpened[identifier] = asset

        if FileManager.default.fileExists(atPath: destination.path) {
            opened = OpenedSplat(id: identifier, title: title, url: destination)
            return
        }
        guard let modelURL = ModelLocator.resolveModelURL() else {
            errorMessage = "No SHARP model found. Run scripts/fetch-model.sh, then choose sharp.mlpackage."
            return
        }

        opened = OpenedSplat(id: identifier, title: title, url: nil)
        sceneProgress = SceneProgress(key: identifier, title: title, stageLabel: "Extracting frame…",
                                      fraction: 0.05, isScene: false, cancellable: false)
        buildStartedAt[identifier] = Date()
        Task { @MainActor in
            guard let avAsset = await self.avAsset(for: asset),
                  let cgImage = await Self.sharpestFrame(from: avAsset) else {
                self.failGeneration(key: identifier, message: "Could not read a frame from the video.")
                return
            }
            self.generate(cgImage: cgImage, identifier: identifier, title: title,
                          modelURL: modelURL, destination: destination)
        }
    }

    /// Shared reconstruction driver used by both photo-group and video scenes.
    /// `prepare` populates `imagesDir` with COLMAP-ready images and returns how
    /// many it wrote, reporting its own 0…1 sub-progress via `report`. Everything
    /// else — tool resolution, the COLMAP+OpenSplat run, progress, caching, and
    /// error handling — is identical. Runs in the background regardless of
    /// navigation; `sceneProgress` tracks it independently of `opened`.
    private func startSceneReconstruction(
        key: String, title: String, options: SceneOptions, hero: PHAsset, sharedCamera: Bool = true,
        prepare: @escaping (_ imagesDir: URL, _ report: @escaping (String, Double) -> Void) async -> Int
    ) {
        let destination = splatURL(for: key)
        if FileManager.default.fileExists(atPath: destination.path) {
            opened = OpenedSplat(id: key, title: title, url: destination, isScene: true,
                                 initialPose: MultiImageReconstructor.initialPose(for: destination))
            return
        }

        guard sceneReconstructor == nil else {
            errorMessage = "A scene is already reconstructing. Wait for it to finish, or cancel it from the progress bar."
            return
        }

        // Resolve the external tools (prompting once if needed).
        guard let colmap = ToolLocator.resolveOrPrompt(for: .colmap),
              let trainer = ToolLocator.resolveOrPrompt(for: .opensplat) else {
            errorMessage = "Multi-image reconstruction needs COLMAP and OpenSplat. "
                + "Run scripts/fetch-tools.sh, then try again."
            return
        }

        // Brush is opt-in. If selected but not resolvable, note it and fall back
        // to OpenSplat rather than blocking the run.
        var brushBinary: URL?
        if options.trainer == .brush {
            brushBinary = ToolLocator.resolveOrPrompt(for: .brush)
            if brushBinary == nil {
                errorMessage = "Brush isn't installed, so OpenSplat was used to train instead. "
                    + "Run scripts/fetch-tools.sh to add it."
            }
        }

        opened = OpenedSplat(id: key, title: title, url: nil, isScene: true)
        sceneProgress = SceneProgress(key: key, title: title, stageLabel: "Preparing…", fraction: 0)
        buildStartedAt[key] = Date()

        let reconstructor = MultiImageReconstructor(colmap: colmap, trainer: trainer)
        reconstructor.trainingIterations = options.iterations
        reconstructor.shDegree = options.shDegree
        // The global solver is COLMAP's own `global_mapper`, so no extra binary to
        // resolve; the reconstructor probes support and falls back if unavailable.
        reconstructor.useGlobalSolver = options.globalPoseSolver
        reconstructor.trainerKind = brushBinary != nil ? .brush : .openSplat
        reconstructor.brushBinary = brushBinary
        sceneReconstructor = reconstructor

        Task.detached(priority: .userInitiated) {
            let fm = FileManager.default
            let workDir = fm.temporaryDirectory.appendingPathComponent("Splatoon/\(key)", isDirectory: true)
            let imagesDir = workDir.appendingPathComponent("photos", isDirectory: true)
            try? fm.removeItem(at: workDir)
            try? fm.createDirectory(at: imagesDir, withIntermediateDirectories: true)

            do {
                // Preparing images occupies the first 5% of the overall bar.
                let written = await prepare(imagesDir) { label, fraction in
                    Task { @MainActor in
                        guard self.sceneProgress?.key == key else { return }
                        self.sceneProgress?.stageLabel = label
                        self.sceneProgress?.fraction = max(0, min(1, fraction)) * 0.05
                    }
                }
                guard written >= 3 else {   // COLMAP needs at least 3 views
                    throw MultiImageReconstructor.ReconstructionError.noSparseModel
                }

                // COLMAP + training: the remaining 95%.
                try reconstructor.run(imagesDir: imagesDir, workDir: workDir, output: destination,
                                      totalImages: written, sharedCamera: sharedCamera) { update in
                    Task { @MainActor in
                        guard self.sceneProgress?.key == key else { return }
                        self.sceneProgress?.stageLabel = update.stageLabel
                        self.sceneProgress?.fraction = 0.05 + 0.95 * update.fraction
                    }
                }

                let initialPose = MultiImageReconstructor.initialPose(for: destination)
                await MainActor.run {
                    self.sceneReconstructor = nil
                    self.splatIdentifiers.insert(key)
                    if self.opened?.id == key {
                        self.opened = OpenedSplat(id: key, title: title, url: destination, isScene: true,
                                                  initialPose: initialPose)
                    }
                    // Leave a brief "Ready" state on the docked bar so a user who
                    // wandered off notices it finished; tapping or a new scene
                    // request clears it (see reopenInProgressScene/dismiss).
                    let elapsed = self.buildStartedAt.removeValue(forKey: key).map(Self.elapsedString(since:))
                    self.sceneProgress = SceneProgress(key: key, title: title, stageLabel: "Ready",
                                                       fraction: 1, isComplete: true, elapsed: elapsed)
                }
                try? fm.removeItem(at: workDir)
            } catch {
                await MainActor.run {
                    self.sceneReconstructor = nil
                    self.sceneProgress = nil
                    if self.opened?.id == key { self.opened = nil }
                    if case MultiImageReconstructor.ReconstructionError.cancelled = error {
                        // User cancelled; stay silent.
                    } else {
                        // Recoverable: offer single-image SHARP on the tapped asset.
                        self.sceneFailure = SceneFailure(message: error.localizedDescription, asset: hero)
                    }
                }
            }
        }
    }

    // MARK: - Frame selection

    /// Turn a mix of photo and video sources into a culled, budgeted set of
    /// COLMAP-ready JPEGs. Videos are frame-sampled and share a global budget so
    /// several clips don't blow up matching; every candidate is quality-scored and
    /// blurred or badly-exposed frames are dropped. Returns the count written.
    private func selectAndWriteFrames(from sources: [PHAsset], to dir: URL,
                                      report: @escaping (String, Double) -> Void) async -> Int {
        let videoCount = sources.filter { $0.mediaType == .video }.count
        let perVideoBudget = videoCount > 0
            ? max(FrameSelection.minFramesPerVideo,
                  min(FrameSelection.maxFramesPerVideo, FrameSelection.totalFrameBudget / videoCount))
            : 0
        let noun = sources.count == 1 ? "source" : "sources"

        var written = 0
        for (sIndex, source) in sources.enumerated() {
            var frames: [CGImage] = []
            if source.mediaType == .video {
                if let av = await avAsset(for: source) {
                    frames = await Self.selectVideoFrames(from: av, budget: perVideoBudget)
                }
            } else if let image = await loadPhoto(source),
                      FrameSelection.exposureUsable(FrameSelection.exposure(of: image)) {
                // A photo is deliberate, so keep it unless it's near-black/blown.
                frames = [image]
            }
            for image in frames {
                let url = dir.appendingPathComponent(String(format: "src%02d_%04d.jpg", sIndex, written))
                if Self.writeJPEG(image, to: url) { written += 1 }
            }
            report("Selecting frames from \(sources.count) \(noun)…",
                   Double(sIndex + 1) / Double(sources.count))
        }
        return written
    }

    /// Resolve a Photos video to an AVAsset.
    private func avAsset(for asset: PHAsset) async -> AVAsset? {
        await withCheckedContinuation { continuation in
            let options = PHVideoRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            imageManager.requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
                continuation.resume(returning: avAsset)
            }
        }
    }

    /// Decode a Photos still, orientation baked in so COLMAP reads it upright.
    private func loadPhoto(_ asset: PHAsset) async -> CGImage? {
        await withCheckedContinuation { (continuation: CheckedContinuation<CGImage?, Never>) in
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            options.isSynchronous = false
            imageManager.requestImageDataAndOrientation(for: asset, options: options) { data, _, _, _ in
                continuation.resume(returning: data.flatMap(Self.decodeOriented))
            }
        }
    }

    /// Sample a video, drop the blurriest and badly-exposed frames, and return up
    /// to `budget` well-exposed frames evenly spread across time (temporal spread
    /// matters more to SfM than picking only the single sharpest moments). A cheap
    /// low-res pass scores; the chosen times are re-rendered at full resolution.
    nonisolated static func selectVideoFrames(from avAsset: AVAsset, budget: Int) async -> [CGImage] {
        guard budget > 0, let duration = try? await avAsset.load(.duration) else { return [] }
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite, seconds > 0 else { return [] }
        let fpsSamples = Int((seconds * FrameSelection.sampleFPS).rounded())
        let sampleCount = min(FrameSelection.maxVideoSamples, max(budget, fpsSamples))

        // Pass 1: cheap low-res scoring to find sharp, well-exposed times.
        let scorer = makeGenerator(avAsset, maxDimension: 480)
        var scored: [(t: Double, sharpness: Double)] = []
        for i in 0..<sampleCount {
            let t = seconds * (Double(i) + 0.5) / Double(sampleCount)
            guard let img = try? await scorer.image(at: CMTime(seconds: t, preferredTimescale: 600)).image,
                  FrameSelection.exposureUsable(FrameSelection.exposure(of: img)) else { continue }
            scored.append((t, FrameSelection.sharpness(of: img)))
        }
        guard !scored.isEmpty else { return [] }

        // Drop frames below a fraction of this clip's median sharpness…
        let median = scored.map(\.sharpness).sorted()[scored.count / 2]
        let ordered = scored.filter { $0.sharpness >= median * FrameSelection.relativeSharpnessFloor }
                            .sorted { $0.t < $1.t }
        // …then evenly subsample the survivors to the budget, preserving spread.
        var times: [Double] = []
        if ordered.count <= budget {
            times = ordered.map(\.t)
        } else {
            let count = Double(ordered.count)
            let slots = Double(budget)
            for k in 0..<budget {
                let position: Double = (Double(k) + 0.5) * count / slots
                let idx = min(Int(position), ordered.count - 1)
                times.append(ordered[idx].t)
            }
        }

        // Pass 2: render the chosen times at full working resolution for COLMAP.
        let full = makeGenerator(avAsset, maxDimension: 1920)
        var frames: [CGImage] = []
        for t in times {
            if let img = try? await full.image(at: CMTime(seconds: t, preferredTimescale: 600)).image {
                frames.append(img)
            }
        }
        return frames
    }

    /// The single sharpest, well-exposed frame of a video, the single-image
    /// fallback's source. Falls back to the midpoint if nothing scores.
    nonisolated static func sharpestFrame(from avAsset: AVAsset) async -> CGImage? {
        guard let duration = try? await avAsset.load(.duration) else { return nil }
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite, seconds > 0 else { return nil }
        let fpsSamples = Int((seconds * FrameSelection.sampleFPS).rounded())
        let sampleCount = min(FrameSelection.maxVideoSamples, max(12, fpsSamples))

        let scorer = makeGenerator(avAsset, maxDimension: 480)
        var best: (t: Double, sharpness: Double)?
        for i in 0..<sampleCount {
            let t = seconds * (Double(i) + 0.5) / Double(sampleCount)
            guard let img = try? await scorer.image(at: CMTime(seconds: t, preferredTimescale: 600)).image,
                  FrameSelection.exposureUsable(FrameSelection.exposure(of: img)) else { continue }
            let s = FrameSelection.sharpness(of: img)
            if best == nil || s > best!.sharpness { best = (t, s) }
        }
        let full = makeGenerator(avAsset, maxDimension: 1920)
        let time = best?.t ?? seconds / 2
        return try? await full.image(at: CMTime(seconds: time, preferredTimescale: 600)).image
    }

    /// Sample evenly-spaced frames from a video into `dir` as JPEGs (no scoring).
    /// Kept for the headless `--selftest-video` hook, which drives sampling from a
    /// file URL without PhotoKit; the interactive path uses `selectVideoFrames`.
    nonisolated static func extractFrames(from avAsset: AVAsset, to dir: URL,
                                          progress: @escaping (Int, Int) -> Void) async -> Int {
        guard let duration = try? await avAsset.load(.duration) else { return 0 }
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite, seconds > 0 else { return 0 }

        let fpsSamples = Int(seconds * FrameSelection.sampleFPS)
        let count = min(FrameSelection.maxFramesPerVideo, max(FrameSelection.minFramesPerVideo, fpsSamples))
        let generator = makeGenerator(avAsset, maxDimension: 1920)
        var written = 0
        for i in 0..<count {
            let t = seconds * (Double(i) + 0.5) / Double(count)
            if let cgImage = try? await generator.image(at: CMTime(seconds: t, preferredTimescale: 600)).image {
                let url = dir.appendingPathComponent(String(format: "frame_%04d.jpg", i))
                if Self.writeJPEG(cgImage, to: url) { written += 1 }
            }
            progress(i + 1, count)
        }
        return written
    }

    /// A frame generator that bakes in orientation, hits exact times, and caps the
    /// output size for speed/memory.
    private nonisolated static func makeGenerator(_ avAsset: AVAsset, maxDimension: CGFloat) -> AVAssetImageGenerator {
        let generator = AVAssetImageGenerator(asset: avAsset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: maxDimension, height: maxDimension)
        return generator
    }

    private nonisolated static func writeJPEG(_ image: CGImage, to url: URL) -> Bool {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else { return false }
        CGImageDestinationAddImage(destination, image,
                                   [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        return CGImageDestinationFinalize(destination)
    }

    private func resetOpenedMesh() {
        openedMesh = nil
        openedMeshKey = nil
    }

    // MARK: - Mesh preview

    /// Build (or reuse) the mesh preview for the opened splat with the given
    /// settings. Reads the cached PLY and meshes it off the main thread, so the
    /// preview always reflects the current meshing code without re-running SHARP.
    func buildOpenedMesh(settingsSignature: String,
                         method: MeshMethod,
                         smoothGrid: Bool,
                         depthRatioCull: Float,
                         surfelExtent: Float,
                         poissonResolution: Int,
                         surfaceTightness: Float,
                         densityOffset: Float) {
        guard let opened, let source = opened.url else { return }
        let key = "\(opened.id)|\(settingsSignature)"
        if openedMeshKey == key, openedMesh != nil { return }   // already current

        // The grid method needs SHARP's structured image grid; unstructured scene
        // splats have none, so mesh them volumetrically instead.
        let method = Self.effectiveMethod(method, isScene: opened.isScene)

        openedMesh = nil
        openedMeshKey = key
        let meshStart = Date()
        let meshTitle = opened.title
        let meshIsScene = opened.isScene
        let meshOwnerID = opened.id
        setIndeterminate("Building mesh…")
        Task.detached(priority: .userInitiated) {
            do {
                let gaussians = try SplatPLYReader.readGaussians(from: source)
                let mesh = MeshExporter.buildMesh(gaussians: gaussians, method: method,
                                                  smoothGrid: smoothGrid, depthRatioCull: depthRatioCull,
                                                  surfelExtent: surfelExtent, poissonResolution: poissonResolution,
                                                  surfaceTightness: surfaceTightness, densityOffset: densityOffset)
                await MainActor.run {
                    guard self.openedMeshKey == key else { return }  // superseded meanwhile
                    self.openedMesh = mesh
                    // Show the mesh compute time in the docked bar (auto-dismisses).
                    self.sceneProgress = SceneProgress(key: meshOwnerID, title: meshTitle,
                                                       stageLabel: "Mesh ready", fraction: 1, isComplete: true,
                                                       isScene: meshIsScene, cancellable: false,
                                                       elapsed: Self.elapsedString(since: meshStart))
                }
            } catch {
                await MainActor.run {
                    guard self.openedMeshKey == key else { return }
                    self.clearIndeterminate()
                    self.errorMessage = "Mesh build failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Indeterminate progress (splat load, mesh build) — routed through
    // the same docked bar as generation, so nothing shows a center overlay.

    /// Show an indeterminate stage in the docked bar for the opened splat. Won't
    /// override an in-flight generation/reconstruction (which owns the bar with a
    /// real fraction).
    private func setIndeterminate(_ stage: String) {
        guard let opened else { return }
        if let p = sceneProgress, p.key == opened.id, !p.indeterminate, !p.isComplete { return }
        sceneProgress = SceneProgress(key: opened.id, title: opened.title, stageLabel: stage,
                                      fraction: 0, isScene: opened.isScene, cancellable: false, indeterminate: true)
    }

    /// Clear an indeterminate stage (leaves determinate/complete progress alone).
    private func clearIndeterminate() {
        if sceneProgress?.indeterminate == true { sceneProgress = nil }
    }

    /// Splat loading (MetalSplatter reading the PLY). Only when a real splat is
    /// open — never during reconstruction (url is nil then).
    func setSplatLoading(_ loading: Bool) {
        guard let opened, opened.url != nil else { return }
        if loading { setIndeterminate("Loading splat…") } else { clearIndeterminate() }
    }

    // MARK: - Cleanup hand-off

    /// Hand the current splat to SuperSplat — PlayCanvas's free, browser-based
    /// splat editor — for cleanup (floater removal, cropping) and publishing.
    /// Serve the local `.ply` over a loopback HTTP server and pass it to
    /// SuperSplat's `?load=<url>` so it opens *with the splat already loaded*,
    /// instead of relying on a manual drag. Falls back to revealing the file if
    /// the server can't start.
    func openInSuperSplat() {
        guard let opened, let source = opened.url else { return }
        Task { @MainActor in
            guard let served = await LocalFileServer.shared.servedURL(for: source),
                  var components = URLComponents(string: "https://superspl.at/editor") else {
                // Fallback: reveal the file so the user can drag it into the editor.
                NSWorkspace.shared.activateFileViewerSelecting([source])
                if let url = URL(string: "https://superspl.at/editor") { NSWorkspace.shared.open(url) }
                return
            }
            components.queryItems = [URLQueryItem(name: "load", value: served.absoluteString)]
            if let url = components.url { NSWorkspace.shared.open(url) }
        }
    }

    // MARK: - Export

    func exportOpened() {
        guard let opened, let source = opened.url else { return }
        let panel = NSSavePanel()
        panel.title = "Export Splat"
        panel.nameFieldStringValue = (opened.title as NSString).deletingPathExtension + ".ply"
        panel.allowedContentTypes = [UTType(filenameExtension: "ply") ?? .data]
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.copyItem(at: source, to: destination)
        } catch {
            errorMessage = "Export failed: \(error.localizedDescription)"
        }
    }

    /// Export a triangle mesh (.glb), built on demand from the cached splat PLY so
    /// it always reflects the current meshing code (no re-inference needed).
    func exportMesh(method: MeshMethod,
                    smoothGrid: Bool,
                    depthRatioCull: Float,
                    surfelExtent: Float,
                    poissonResolution: Int,
                    surfaceTightness: Float,
                    densityOffset: Float) {
        guard let opened, let source = opened.url else { return }
        let panel = NSSavePanel()
        panel.title = "Export Mesh"
        panel.nameFieldStringValue = (opened.title as NSString).deletingPathExtension + ".glb"
        panel.allowedContentTypes = [UTType(filenameExtension: "glb") ?? .data]
        guard panel.runModal() == .OK, let destination = panel.url else { return }

        let method = Self.effectiveMethod(method, isScene: opened.isScene)
        setIndeterminate("Exporting mesh…")
        Task.detached(priority: .userInitiated) {
            do {
                let gaussians = try SplatPLYReader.readGaussians(from: source)
                try MeshExporter.saveGLB(gaussians: gaussians, to: destination,
                                         method: method, smoothGrid: smoothGrid,
                                         depthRatioCull: depthRatioCull, surfelExtent: surfelExtent,
                                         poissonResolution: poissonResolution,
                                         surfaceTightness: surfaceTightness, densityOffset: densityOffset)
                await MainActor.run { self.clearIndeterminate() }
            } catch {
                await MainActor.run {
                    self.clearIndeterminate()
                    self.errorMessage = "Mesh export failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Helpers

    /// The grid method assumes SHARP's structured image grid. Scene splats are
    /// unstructured, so fall back to volumetric (poisson) meshing for them.
    private static func effectiveMethod(_ method: MeshMethod, isScene: Bool) -> MeshMethod {
        (isScene && method == .grid) ? .poisson : method
    }

    private static func title(for asset: PHAsset) -> String {
        PHAssetResource.assetResources(for: asset).first?.originalFilename ?? "Splat"
    }

    /// Decodes image data, baking in EXIF orientation so portrait photos are
    /// upright for the model.
    private nonisolated static func decodeOriented(_ data: Data) -> CGImage? {
        autoreleasepool {   // drain the large intermediate CG/CI buffers promptly
            guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let raw = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
            let rawOrientation = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
            guard rawOrientation != 1,
                  let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) else {
                return raw
            }
            let ciImage = CIImage(cgImage: raw).oriented(orientation)
            return sharedCIContext.createCGImage(ciImage, from: ciImage.extent) ?? raw
        }
    }
}
