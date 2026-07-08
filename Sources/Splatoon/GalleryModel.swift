import SwiftUI
import AppKit
import Photos
import ImageIO
import CoreImage
import CoreGraphics
import UniformTypeIdentifiers

/// Drives the Photos-library gallery: authorization, asset listing, thumbnail
/// image manager, per-image splat generation, and an on-disk splat cache.
@MainActor
final class GalleryModel: ObservableObject {

    struct OpenedSplat: Identifiable, Equatable {
        let id: String        // PHAsset.localIdentifier, or a scene key
        let title: String
        let url: URL?         // nil while the splat is still being generated
        var isScene = false   // reconstructed from multiple photos (unstructured)
    }

    @Published private(set) var authorization: PHAuthorizationStatus =
        PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @Published private(set) var assets: [PHAsset] = []
    @Published private(set) var splatIdentifiers: Set<String> = []
    @Published var opened: OpenedSplat?
    @Published private(set) var busyMessage: String?
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
        options.predicate = NSPredicate(format: "mediaType == %d", PHAssetMediaType.image.rawValue)
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

    func open(_ asset: PHAsset) {
        errorMessage = nil
        resetOpenedMesh()

        // If this photo is one of several capturing the same place/moment, use
        // them together (multi-view). Otherwise fall through to single-image SHARP.
        let group = SceneGrouping.neighbors(of: asset, in: assets)
        if SceneGrouping.isScene(group) {
            openScene(group)
            return
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

        // Switch to the splat view immediately (url nil = still generating) so
        // progress is shown there rather than over the gallery.
        opened = OpenedSplat(id: identifier, title: title, url: nil)
        busyMessage = "Loading photo…"
        let requestOptions = PHImageRequestOptions()
        requestOptions.isNetworkAccessAllowed = true
        requestOptions.deliveryMode = .highQualityFormat
        requestOptions.isSynchronous = false

        imageManager.requestImageDataAndOrientation(for: asset, options: requestOptions) { [weak self] data, _, _, _ in
            Task { @MainActor in
                guard let self else { return }
                guard let data, let cgImage = Self.decodeOriented(data) else {
                    self.busyMessage = nil
                    self.errorMessage = "Could not load that photo."
                    if self.opened?.id == identifier { self.opened = nil }
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
        busyMessage = "Loading model…"
        let decimation = self.decimation
        let existingRunner = (cachedRunnerURL == modelURL) ? cachedRunner : nil

        Task.detached(priority: .userInitiated) {
            do {
                let runner = try existingRunner ?? SharpModelRunner(modelURL: modelURL)
                await MainActor.run {
                    self.cachedRunner = runner
                    self.cachedRunnerURL = modelURL
                    self.busyMessage = "Preparing image…"
                }
                let input = try runner.preprocess(cgImage)

                await MainActor.run { self.busyMessage = "Running inference…" }
                let gaussians = try runner.predict(image: input, focalLengthPx: 1536)

                await MainActor.run { self.busyMessage = "Writing splat…" }
                try SplatExporter.savePLY(
                    gaussians: gaussians,
                    focalLengthPx: 1536,
                    imageShape: (height: runner.inputHeight, width: runner.inputWidth),
                    to: destination,
                    decimation: decimation
                )

                await MainActor.run {
                    self.splatIdentifiers.insert(identifier)
                    self.busyMessage = nil
                    // Only reveal the splat if the user is still on this one
                    // (they may have navigated back during generation).
                    if self.opened?.id == identifier {
                        self.opened = OpenedSplat(id: identifier, title: title, url: destination)
                    }
                }
            } catch {
                await MainActor.run {
                    self.busyMessage = nil
                    self.errorMessage = "Generation failed: \(error.localizedDescription)"
                    if self.opened?.id == identifier { self.opened = nil }
                }
            }
        }
    }

    func closeOpened() {
        opened = nil
        resetOpenedMesh()
        sceneReconstructor?.cancel()   // stop any in-flight reconstruction
        sceneReconstructor = nil
    }

    // MARK: - Multi-image scene reconstruction

    /// Reconstruct a splat from several photos of the same scene (COLMAP +
    /// OpenSplat), caching the result like a single-image splat.
    private func openScene(_ group: [PHAsset]) {
        let key = SceneGrouping.sceneKey(for: group)
        let destination = splatURL(for: key)
        let title = "Scene · \(group.count) photos"

        if FileManager.default.fileExists(atPath: destination.path) {
            opened = OpenedSplat(id: key, title: title, url: destination, isScene: true)
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
        busyMessage = "Found \(group.count) related photos — reconstructing scene…"

        let reconstructor = MultiImageReconstructor(colmap: colmap, trainer: trainer)
        sceneReconstructor = reconstructor
        let assets = group

        Task.detached(priority: .userInitiated) {
            let fm = FileManager.default
            let workDir = fm.temporaryDirectory.appendingPathComponent("Splatoon/\(key)", isDirectory: true)
            let imagesDir = workDir.appendingPathComponent("photos", isDirectory: true)
            try? fm.removeItem(at: workDir)
            try? fm.createDirectory(at: imagesDir, withIntermediateDirectories: true)

            do {
                await MainActor.run { self.busyMessage = "Preparing \(assets.count) photos…" }
                var written = 0
                for (index, asset) in assets.enumerated() {
                    if await self.exportSceneImage(asset, to: imagesDir, index: index) { written += 1 }
                }
                guard written >= SceneGrouping.minSceneSize else {
                    throw MultiImageReconstructor.ReconstructionError.noSparseModel
                }

                try reconstructor.run(imagesDir: imagesDir, workDir: workDir, output: destination) { message in
                    Task { @MainActor in
                        if self.opened?.id == key { self.busyMessage = message }
                    }
                }

                await MainActor.run {
                    self.sceneReconstructor = nil
                    self.splatIdentifiers.insert(key)
                    self.busyMessage = nil
                    if self.opened?.id == key {
                        self.opened = OpenedSplat(id: key, title: title, url: destination, isScene: true)
                    }
                }
                try? fm.removeItem(at: workDir)
            } catch {
                await MainActor.run {
                    self.sceneReconstructor = nil
                    self.busyMessage = nil
                    if case MultiImageReconstructor.ReconstructionError.cancelled = error {
                        // User navigated back mid-run; stay silent.
                    } else {
                        self.errorMessage = "Scene reconstruction failed: \(error.localizedDescription)"
                        if self.opened?.id == key { self.opened = nil }
                    }
                }
            }
        }
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
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let raw = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let rawOrientation = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        guard rawOrientation != 1,
              let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) else {
            return raw
        }
        let ciImage = CIImage(cgImage: raw).oriented(orientation)
        return CIContext(options: nil).createCGImage(ciImage, from: ciImage.extent) ?? raw
    }
}
