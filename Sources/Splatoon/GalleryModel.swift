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
    @Published private(set) var busyMessage: String?
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

    /// Open an asset. A **video** is reconstructed as a multi-view scene from
    /// frames sampled across it (dense, overlapping frames are ideal SfM input).
    /// A **photo** with enough same-place/time siblings likewise reconstructs as
    /// a scene; a lone photo goes through single-image SHARP. Scenes are gated by
    /// `allowMultiImage` (the "Reconstruct multi-image scenes" setting).
    func open(_ asset: PHAsset, allowMultiImage: Bool = true, multiImageIterations: Int = 15000) {
        errorMessage = nil
        resetOpenedMesh()

        if asset.mediaType == .video {
            guard allowMultiImage else {
                errorMessage = "Turn on “Reconstruct multi-image scenes” in Settings (⌘,) to build a splat from a video."
                return
            }
            openVideoScene(asset, iterations: multiImageIterations)
            return
        }

        if allowMultiImage {
            let group = SceneGrouping.neighbors(of: asset, in: assets)
            if SceneGrouping.isScene(group) {
                openScene(group, hero: asset, iterations: multiImageIterations)
                return
            }
        }

        let identifier = asset.localIdentifier
        let destination = splatURL(for: identifier)
        let title = Self.title(for: asset)

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
        sceneProgress = SceneProgress(key: key, title: title, stageLabel: "Ready", fraction: 1,
                                      isComplete: true, isScene: isScene, cancellable: false)
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

    /// Reconstruct a splat from several photos of the same scene, caching the
    /// result like a single-image splat.
    private func openScene(_ group: [PHAsset], hero: PHAsset, iterations: Int) {
        // Iteration count is part of the cache key so raising it in Settings and
        // reopening actually retrains, instead of silently reusing a blurrier cache.
        let key = SceneGrouping.sceneKey(for: group) + "-i\(iterations)"
        startSceneReconstruction(key: key, title: "Scene · \(group.count) photos",
                                 iterations: iterations, hero: hero) { imagesDir, report in
            var written = 0
            for (index, asset) in group.enumerated() {
                if await self.exportSceneImage(asset, to: imagesDir, index: index) { written += 1 }
                report("Preparing \(group.count) photos…", Double(index + 1) / Double(group.count))
            }
            return written
        }
    }

    /// Reconstruct a splat from a single video, sampling frames across it — dense,
    /// overlapping frames are ideal SfM input, sidestepping the sparse-photo
    /// registration failures.
    private func openVideoScene(_ asset: PHAsset, iterations: Int) {
        let key = "video-" + SceneGrouping.stableHash(asset.localIdentifier) + "-i\(iterations)"
        startSceneReconstruction(key: key, title: "Video scene", iterations: iterations, hero: asset) { imagesDir, report in
            await self.extractVideoFrames(from: asset, to: imagesDir) { done, total in
                report("Extracting \(total) frames…", Double(done) / Double(max(total, 1)))
            }
        }
    }

    /// Recover from a failed scene by running single-image SHARP on the asset the
    /// user tapped (its middle frame, for a video). Driven by the failure alert.
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

    /// SHARP on a video's middle frame — the single-image fallback for a video.
    private func openVideoSingleFrame(_ asset: PHAsset) {
        resetOpenedMesh()
        let identifier = asset.localIdentifier
        let destination = splatURL(for: identifier)
        let title = Self.title(for: asset)

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
        Task { @MainActor in
            guard let cgImage = await self.middleVideoFrame(from: asset) else {
                self.failGeneration(key: identifier, message: "Could not read a frame from the video.")
                return
            }
            self.generate(cgImage: cgImage, identifier: identifier, title: title,
                          modelURL: modelURL, destination: destination)
        }
    }

    /// Decode the frame at the midpoint of a Photos video, orientation baked in.
    private func middleVideoFrame(from asset: PHAsset) async -> CGImage? {
        let avAsset: AVAsset? = await withCheckedContinuation { continuation in
            let options = PHVideoRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            imageManager.requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
                continuation.resume(returning: avAsset)
            }
        }
        guard let avAsset, let duration = try? await avAsset.load(.duration) else { return nil }
        let generator = AVAssetImageGenerator(asset: avAsset)
        generator.appliesPreferredTrackTransform = true
        let mid = CMTime(seconds: CMTimeGetSeconds(duration) / 2, preferredTimescale: 600)
        return try? await generator.image(at: mid).image
    }

    /// Shared reconstruction driver used by both photo-group and video scenes.
    /// `prepare` populates `imagesDir` with COLMAP-ready images and returns how
    /// many it wrote, reporting its own 0…1 sub-progress via `report`. Everything
    /// else — tool resolution, the COLMAP+OpenSplat run, progress, caching, and
    /// error handling — is identical. Runs in the background regardless of
    /// navigation; `sceneProgress` tracks it independently of `opened`.
    private func startSceneReconstruction(
        key: String, title: String, iterations: Int, hero: PHAsset,
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

        opened = OpenedSplat(id: key, title: title, url: nil, isScene: true)
        sceneProgress = SceneProgress(key: key, title: title, stageLabel: "Preparing…", fraction: 0)

        let reconstructor = MultiImageReconstructor(colmap: colmap, trainer: trainer)
        reconstructor.trainingIterations = iterations
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
                                      totalImages: written) { update in
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
                    self.sceneProgress = SceneProgress(key: key, title: title, stageLabel: "Ready",
                                                       fraction: 1, isComplete: true)
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

    /// Resolve the Photos video to an AVAsset and sample frames from it (see
    /// `extractFrames`).
    private func extractVideoFrames(from asset: PHAsset, to dir: URL,
                                    progress: @escaping (Int, Int) -> Void) async -> Int {
        let avAsset: AVAsset? = await withCheckedContinuation { continuation in
            let options = PHVideoRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            imageManager.requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
                continuation.resume(returning: avAsset)
            }
        }
        guard let avAsset else { return 0 }
        return await Self.extractFrames(from: avAsset, to: dir, progress: progress)
    }

    /// Sample evenly-spaced frames from a video into `dir` as JPEGs (orientation
    /// baked in, downscaled for speed). ~3 fps, clamped 15…48 frames — dense
    /// enough to register reliably without O(n²) matching exploding. Returns the
    /// count written; `progress(done, total)` reports extraction progress.
    /// `nonisolated static` so the headless `--selftest-video` hook can drive the
    /// exact same sampling from a file URL, without PhotoKit.
    nonisolated static func extractFrames(from avAsset: AVAsset, to dir: URL,
                                          progress: @escaping (Int, Int) -> Void) async -> Int {
        guard let duration = try? await avAsset.load(.duration) else { return 0 }
        let seconds = CMTimeGetSeconds(duration)
        guard seconds.isFinite, seconds > 0 else { return 0 }

        let count = min(48, max(15, Int(seconds * 3)))
        let generator = AVAssetImageGenerator(asset: avAsset)
        generator.appliesPreferredTrackTransform = true            // bake in orientation
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: 1920, height: 1920)  // downscale for speed

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

    /// Export one Photos asset to a JPEG in `dir`, orientation baked in so COLMAP
    /// reads it upright. Returns whether the write succeeded.
    private func exportSceneImage(_ asset: PHAsset, to dir: URL, index: Int) async -> Bool {
        await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            options.isSynchronous = false
            imageManager.requestImageDataAndOrientation(for: asset, options: options) { data, _, _, _ in
                guard let data, let cgImage = Self.decodeOriented(data) else {
                    continuation.resume(returning: false)
                    return
                }
                let url = dir.appendingPathComponent(String(format: "img_%04d.jpg", index))
                continuation.resume(returning: Self.writeJPEG(cgImage, to: url))
            }
        }
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
                         poissonResolution: Int) {
        guard let opened, let source = opened.url else { return }
        let key = "\(opened.id)|\(settingsSignature)"
        if openedMeshKey == key, openedMesh != nil { return }   // already current

        // The grid method needs SHARP's structured image grid; unstructured scene
        // splats have none, so mesh them volumetrically instead.
        let method = Self.effectiveMethod(method, isScene: opened.isScene)

        openedMesh = nil
        openedMeshKey = key
        busyMessage = "Building mesh…"
        Task.detached(priority: .userInitiated) {
            do {
                let gaussians = try SplatPLYReader.readGaussians(from: source)
                let mesh = MeshExporter.buildMesh(gaussians: gaussians, method: method,
                                                  smoothGrid: smoothGrid, depthRatioCull: depthRatioCull,
                                                  surfelExtent: surfelExtent, poissonResolution: poissonResolution)
                await MainActor.run {
                    guard self.openedMeshKey == key else { return }  // superseded meanwhile
                    self.openedMesh = mesh
                    self.busyMessage = nil
                }
            } catch {
                await MainActor.run {
                    guard self.openedMeshKey == key else { return }
                    self.busyMessage = nil
                    self.errorMessage = "Mesh build failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Cleanup hand-off

    /// Hand the current splat to SuperSplat — PlayCanvas's free, browser-based
    /// splat editor — for cleanup (floater removal, cropping) and publishing. It
    /// processes splats client-side and can't auto-load a local file, so reveal
    /// the cached `.ply` in Finder and open the editor; the user drags it in.
    func openInSuperSplat() {
        guard let opened, let source = opened.url else { return }
        NSWorkspace.shared.activateFileViewerSelecting([source])
        if let url = URL(string: "https://superspl.at/editor") {
            NSWorkspace.shared.open(url)
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
                    poissonResolution: Int) {
        guard let opened, let source = opened.url else { return }
        let panel = NSSavePanel()
        panel.title = "Export Mesh"
        panel.nameFieldStringValue = (opened.title as NSString).deletingPathExtension + ".glb"
        panel.allowedContentTypes = [UTType(filenameExtension: "glb") ?? .data]
        guard panel.runModal() == .OK, let destination = panel.url else { return }

        let method = Self.effectiveMethod(method, isScene: opened.isScene)
        busyMessage = "Building mesh…"
        Task.detached(priority: .userInitiated) {
            do {
                let gaussians = try SplatPLYReader.readGaussians(from: source)
                try MeshExporter.saveGLB(gaussians: gaussians, to: destination,
                                         method: method, smoothGrid: smoothGrid,
                                         depthRatioCull: depthRatioCull, surfelExtent: surfelExtent,
                                         poissonResolution: poissonResolution)
                await MainActor.run { self.busyMessage = nil }
            } catch {
                await MainActor.run {
                    self.busyMessage = nil
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
