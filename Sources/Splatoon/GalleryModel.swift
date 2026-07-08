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
        let id: String        // PHAsset.localIdentifier
        let title: String
        let url: URL?         // nil while the splat is still being generated
    }

    @Published private(set) var authorization: PHAuthorizationStatus =
        PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @Published private(set) var assets: [PHAsset] = []
    @Published private(set) var splatIdentifiers: Set<String> = []
    @Published var opened: OpenedSplat?
    @Published private(set) var busyMessage: String?
    @Published var errorMessage: String?

    /// Fraction of Gaussians to keep (1.0 = all).
    var decimation: Float = 1.0

    let imageManager = PHCachingImageManager()

    private let cacheDir: URL
    private var cachedRunner: SharpModelRunner?
    private var cachedRunnerURL: URL?

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
