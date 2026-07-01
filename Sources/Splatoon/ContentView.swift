import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var model = AppModel()
    @State private var showFileImporter = false

    var body: some View {
        HSplitView {
            controls
                .frame(minWidth: 260, idealWidth: 300, maxWidth: 340)
            preview
                .frame(minWidth: 480)
        }
    }

    // MARK: - Controls

    private var controls: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Splatoon").font(.largeTitle.bold())
                Text("Turn a photo into a 3D Gaussian splat, on-device.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            PhotosPicker(selection: $model.photoSelection,
                         matching: .images,
                         photoLibrary: .shared()) {
                Label("Choose from Photos", systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Button {
                showFileImporter = true
            } label: {
                Label("Open Image File…", systemImage: "folder")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Divider()

            VStack(alignment: .leading, spacing: 4) {
                Text("Detail: \(Int(model.decimation * 100))% of Gaussians")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Slider(value: $model.decimation, in: 0.1...1.0)
            }
            .disabled(isGenerating)

            Button {
                model.generateSplat()
            } label: {
                Label("Generate Splat", systemImage: "cube.transparent")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.hasImage || isGenerating)

            if model.generatedSplat != nil {
                Button {
                    model.exportSplat()
                } label: {
                    Label("Export Splat…", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            statusView

            Spacer()
        }
        .padding()
        .fileImporter(isPresented: $showFileImporter,
                      allowedContentTypes: [.image]) { result in
            switch result {
            case .success(let url):
                model.loadFromFile(url)
            case .failure(let error):
                // Surface the OS error; the model reports its own load errors.
                print("File import failed: \(error.localizedDescription)")
            }
        }
    }

    private var isGenerating: Bool {
        if case .generating = model.stage { return true }
        return false
    }

    @ViewBuilder
    private var statusView: some View {
        switch model.stage {
        case .idle:
            EmptyView()
        case .loadingImage:
            ProgressView("Loading image…")
        case .ready:
            if let count = model.lastGaussianCount, model.generatedSplat != nil {
                Label("Generated \(count.formatted()) Gaussians",
                      systemImage: "checkmark.circle")
                    .foregroundStyle(.green)
                    .font(.callout)
            } else {
                Label("Image ready", systemImage: "checkmark.circle")
                    .foregroundStyle(.green)
            }
        case .generating(let message):
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(message)
            }
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
                .font(.callout)
        }
    }

    // MARK: - Preview

    @ViewBuilder
    private var preview: some View {
        if let splat = model.generatedSplat {
            VStack(spacing: 0) {
                SplatViewer(url: splat)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                Text("Drag to orbit · scroll or pinch to zoom")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(6)
            }
        } else if let image = model.displayImage {
            Image(nsImage: image)
                .resizable()
                .scaledToFit()
                .padding()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(.black.opacity(0.03))
        } else {
            ContentUnavailableView(
                "No image selected",
                systemImage: "photo",
                description: Text("Choose a photo or open an image file to begin.")
            )
        }
    }
}

#Preview {
    ContentView()
}
