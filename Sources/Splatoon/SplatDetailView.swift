import SwiftUI

/// Full-window splat viewer shown after opening an image, with a back button
/// and export.
struct SplatDetailView: View {
    @ObservedObject var model: GalleryModel
    let opened: GalleryModel.OpenedSplat
    @State private var isLoading = true

    var body: some View {
        SplatViewer(url: opened.url, onLoadingChange: { isLoading = $0 })
            .ignoresSafeArea()
            .overlay { if let text = progressText { progressOverlay(text) } }
            .overlay(alignment: .top) { toolbar }
            .overlay(alignment: .bottom) { if progressText == nil { hint } }
            .animation(.easeInOut(duration: 0.2), value: progressText)
    }

    /// The progress message to show over the splat view, or nil when the splat is
    /// ready and interactive. Covers generation, mesh export, and renderer load.
    private var progressText: String? {
        if let busy = model.busyMessage { return busy }        // generating / building mesh
        if opened.url == nil { return "Generating splat…" }     // switched in before any message
        if isLoading { return "Loading splat…" }                // reading the PLY into the renderer
        return nil
    }

    private func progressOverlay(_ text: String) -> some View {
        ZStack {
            Color.black.opacity(0.35).ignoresSafeArea()
            VStack(spacing: 14) {
                ProgressView().controlSize(.large)
                Text(text).foregroundStyle(.white)
            }
            .padding(28)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
        .allowsHitTesting(false)   // keep the toolbar (back) live during progress
        .transition(.opacity)
    }

    private var toolbar: some View {
        HStack {
            Button {
                model.closeOpened()
            } label: {
                Label("Gallery", systemImage: "chevron.backward")
            }

            Spacer()

            Text(opened.title)
                .font(.headline)
                .lineLimit(1)

            Spacer()

            Menu {
                Button("Splat (.ply)") { model.exportOpened() }
                Button("Mesh (.glb)") { model.exportMesh() }
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .disabled(opened.url == nil)
        }
        .padding(12)
        .background(.ultraThinMaterial)
    }

    private var hint: some View {
        Text("Drag to look · WASD to move · Q/E up-down · scroll to dolly · R to reset")
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(8)
            .background(.ultraThinMaterial, in: Capsule())
            .padding(.bottom, 12)
    }
}
