import SwiftUI

/// Full-window splat viewer shown after opening an image, with a back button
/// and export.
struct SplatDetailView: View {
    @ObservedObject var model: GalleryModel
    let opened: GalleryModel.OpenedSplat

    var body: some View {
        SplatViewer(url: opened.url)
            .ignoresSafeArea()
            .overlay(alignment: .top) { toolbar }
            .overlay(alignment: .bottom) { hint }
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

            Button {
                model.exportOpened()
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
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
