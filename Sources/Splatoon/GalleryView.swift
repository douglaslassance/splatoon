import SwiftUI
import Photos

struct GalleryView: View {
    @ObservedObject var model: GalleryModel
    @EnvironmentObject var settings: MeshSettings

    private let columns = [GridItem(.adaptive(minimum: 130), spacing: 2)]

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 2) {
                ForEach(model.assets, id: \.localIdentifier) { asset in
                    ThumbnailCell(asset: asset,
                                  imageManager: model.imageManager,
                                  hasSplat: model.hasSplat(asset))
                        .onTapGesture(count: 2) {
                            model.open(asset, allowMultiImage: settings.useMultiImageReconstruction,
                                      multiImageIterations: Int(settings.multiImageIterations))
                        }
                        .help("Double-click to open its 3D splat")
                }
            }
            .padding(2)
        }
        .background(.background)
    }
}

/// A single square gallery cell that lazily loads its Photos thumbnail and marks
/// images that already have a generated splat.
struct ThumbnailCell: View {
    let asset: PHAsset
    let imageManager: PHCachingImageManager
    let hasSplat: Bool

    @State private var image: NSImage?
    @State private var isHovering = false

    var body: some View {
        Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Rectangle().fill(.quaternary)
                }
            }
            .clipped()
            .overlay(alignment: .bottomTrailing) {
                if hasSplat {
                    Image(systemName: "cube.fill")
                        .font(.caption)
                        .foregroundStyle(.white)
                        .padding(5)
                        .background(.black.opacity(0.55), in: Circle())
                        .padding(5)
                }
            }
            .overlay {
                if isHovering {
                    Rectangle().stroke(Color.accentColor, lineWidth: 3)
                }
            }
            .contentShape(Rectangle())
            .onHover { isHovering = $0 }
            .task(id: asset.localIdentifier) { requestThumbnail() }
    }

    private func requestThumbnail() {
        let options = PHImageRequestOptions()
        options.deliveryMode = .opportunistic
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true
        let scale = NSScreen.main?.backingScaleFactor ?? 2
        let target = CGSize(width: 260 * scale, height: 260 * scale)
        imageManager.requestImage(for: asset,
                                  targetSize: target,
                                  contentMode: .aspectFill,
                                  options: options) { result, _ in
            if let result {
                Task { @MainActor in self.image = result }
            }
        }
    }
}
