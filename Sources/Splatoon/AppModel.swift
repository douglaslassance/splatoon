import SwiftUI
import PhotosUI
import ImageIO
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
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw LoadError.decodeFailed
        }
        inputImage = image
        stage = .ready
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
