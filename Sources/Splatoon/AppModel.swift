import SwiftUI
import AppKit
import PhotosUI
import ImageIO
import CoreImage
import UniformTypeIdentifiers

/// Owns the app's input image and pipeline state. Phase 1 covers image
/// selection (Photos library + file). Later phases add Core ML inference,
/// splat export, and the in-app viewer.
@MainActor
final class AppModel: ObservableObject {

    enum Stage: Equatable {
        case idle
        case loadingImage
        case ready
        case generating(String)
        case failed(String)
    }

    @Published private(set) var inputImage: CGImage?
    @Published private(set) var stage: Stage = .idle

    /// URL of the most recently generated splat (a PLY in a temp directory).
    @Published private(set) var generatedSplat: URL?
    @Published private(set) var lastGaussianCount: Int?

    /// Focal length in pixels; the model derives its disparity factor from this
    /// (`focal / inputWidth`). 1536 matches the training resolution → factor 1.0.
    var focalLengthPx: Float = 1536
    /// Fraction of Gaussians to keep (1.0 = all, 0.5 = most important 50%).
    var decimation: Float = 1.0

    /// The loaded model, kept resident and reused across generations.
    private var cachedRunner: SharpModelRunner?
    private var cachedRunnerURL: URL?

    /// Bound to the `PhotosPicker`. Selecting an item kicks off a load.
    @Published var photoSelection: PhotosPickerItem? {
        didSet {
            guard let photoSelection else { return }
            loadFromPhotos(photoSelection)
        }
    }

    /// An `NSImage` wrapper for SwiftUI display; the `CGImage` remains the
    /// source of truth for downstream image processing.
    var displayImage: NSImage? {
        guard let inputImage else { return nil }
        return NSImage(cgImage: inputImage,
                       size: NSSize(width: inputImage.width, height: inputImage.height))
    }

    var hasImage: Bool { inputImage != nil }

    // MARK: - Loading

    func loadFromPhotos(_ item: PhotosPickerItem) {
        stage = .loadingImage
        Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self) else {
                    throw LoadError.noData
                }
                try setImage(from: data)
            } catch {
                stage = .failed("Could not load photo: \(error.localizedDescription)")
            }
        }
    }

    func loadFromFile(_ url: URL) {
        stage = .loadingImage
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let data = try Data(contentsOf: url)
            try setImage(from: data)
        } catch {
            stage = .failed("Could not open file: \(error.localizedDescription)")
        }
    }

    private func setImage(from data: Data) throws {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let raw = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw LoadError.decodeFailed
        }
        inputImage = Self.orientationCorrected(raw, source: source)
        generatedSplat = nil
        lastGaussianCount = nil
        stage = .ready
        // Picking an image immediately kicks off generation.
        generateSplat()
    }

    /// Bakes the source's EXIF orientation into the pixels so portrait photos
    /// (stored landscape + a rotation flag) are upright for both display and the
    /// model. `CGImageSourceCreateImageAtIndex` ignores the orientation tag.
    private nonisolated static func orientationCorrected(_ cgImage: CGImage,
                                                         source: CGImageSource) -> CGImage {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let rawOrientation = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        guard rawOrientation != 1,
              let orientation = CGImagePropertyOrientation(rawValue: rawOrientation) else {
            return cgImage
        }
        let ciImage = CIImage(cgImage: cgImage).oriented(orientation)
        let context = CIContext(options: nil)
        return context.createCGImage(ciImage, from: ciImage.extent) ?? cgImage
    }

    // MARK: - Generation

    func generateSplat() {
        guard let cgImage = inputImage else { return }

        // Resolving the model may present an open panel, so do it on the main
        // actor before handing off to background compute.
        guard let modelURL = ModelLocator.resolveModelURL() else {
            stage = .failed("No SHARP model selected. Run scripts/fetch-model.sh, then choose sharp.mlpackage.")
            return
        }

        stage = .generating("Loading model…")
        let focal = focalLengthPx
        let decimation = self.decimation
        // Reuse the loaded model across generations. This avoids reloading the
        // ~2.5 GB model each time and, crucially, keeps it alive so its Core ML
        // GPU resources aren't torn down on a background thread while the splat
        // renderer is initializing.
        let existingRunner = (cachedRunnerURL == modelURL) ? cachedRunner : nil

        Task.detached(priority: .userInitiated) {
            do {
                let runner = try existingRunner ?? SharpModelRunner(modelURL: modelURL)
                await MainActor.run {
                    self.cachedRunner = runner
                    self.cachedRunnerURL = modelURL
                }

                await MainActor.run { self.stage = .generating("Preparing image…") }
                let input = try runner.preprocess(cgImage)

                await MainActor.run { self.stage = .generating("Running inference…") }
                let gaussians = try runner.predict(image: input, focalLengthPx: focal)
                let count = gaussians.count

                await MainActor.run { self.stage = .generating("Writing splat…") }
                let outputURL = Self.makeOutputURL()
                try SplatExporter.savePLY(
                    gaussians: gaussians,
                    focalLengthPx: focal,
                    imageShape: (height: runner.inputHeight, width: runner.inputWidth),
                    to: outputURL,
                    decimation: decimation
                )

                await MainActor.run {
                    self.generatedSplat = outputURL
                    self.lastGaussianCount = count
                    self.stage = .ready
                }
            } catch {
                await MainActor.run {
                    self.stage = .failed("Generation failed: \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: - Export

    func exportSplat() {
        guard let source = generatedSplat else { return }
        let panel = NSSavePanel()
        panel.title = "Export Splat"
        panel.nameFieldStringValue = "splat.ply"
        panel.allowedContentTypes = [UTType(filenameExtension: "ply") ?? .data]
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.copyItem(at: source, to: destination)
        } catch {
            stage = .failed("Export failed: \(error.localizedDescription)")
        }
    }

    private nonisolated static func makeOutputURL() -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("SplatoonOutputs", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("splat-\(UUID().uuidString).ply")
    }

    enum LoadError: LocalizedError {
        case noData
        case decodeFailed

        var errorDescription: String? {
            switch self {
            case .noData: return "No image data was returned."
            case .decodeFailed: return "The image could not be decoded."
            }
        }
    }
}
