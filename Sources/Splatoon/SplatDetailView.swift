import SwiftUI

/// Full-window viewer shown after opening an image. Toggles between the Gaussian
/// splat and its triangle-mesh preview, with a back button and a context-aware
/// export that follows whichever view is showing.
struct SplatDetailView: View {
    @ObservedObject var model: GalleryModel
    @EnvironmentObject var settings: MeshSettings
    let opened: GalleryModel.OpenedSplat
    @State private var isLoading = true
    @State private var viewMode: ViewMode = .splat

    enum ViewMode: String { case splat, mesh }

    var body: some View {
        content
            .ignoresSafeArea()
            .overlay { if let text = progressText { progressOverlay(text) } }
            .overlay(alignment: .top) { toolbar }
            .overlay(alignment: .bottom) { if progressText == nil { hint } }
            .animation(.easeInOut(duration: 0.2), value: progressText)
            // Rebuild the mesh preview when entering mesh mode, changing the opened
            // splat, or tweaking mesh settings. `.task(id:)` cancels/reruns on change.
            .task(id: meshTaskID) {
                guard viewMode == .mesh, opened.url != nil else { return }
                model.buildOpenedMesh(settingsSignature: settings.signature,
                                      method: settings.method,
                                      smoothGrid: settings.smoothGrid,
                                      depthRatioCull: Float(settings.depthRatioCull),
                                      surfelExtent: Float(settings.surfelExtent),
                                      poissonResolution: Int(settings.poissonResolution))
            }
    }

    @ViewBuilder
    private var content: some View {
        switch viewMode {
        case .splat:
            SplatViewer(url: opened.url, onLoadingChange: { isLoading = $0 })
        case .mesh:
            if let mesh = model.openedMesh {
                MeshViewer(mesh: mesh)
            } else {
                Color.clear   // progress overlay covers the build
            }
        }
    }

    /// Identity for the mesh-rebuild task: mode, splat, availability, and settings.
    private var meshTaskID: String {
        "\(viewMode.rawValue)|\(opened.id)|\(opened.url != nil)|\(settings.signature)"
    }

    /// The progress message to show over the view, or nil when it's interactive.
    private var progressText: String? {
        if let busy = model.busyMessage { return busy }        // generating / building
        if opened.url == nil { return "Generating splat…" }     // still inferring
        switch viewMode {
        case .splat:
            return isLoading ? "Loading splat…" : nil
        case .mesh:
            return model.openedMesh == nil ? "Building mesh…" : nil
        }
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
        HStack(spacing: 12) {
            Button {
                model.closeOpened()
            } label: {
                Label("Gallery", systemImage: "chevron.backward")
            }

            Text(opened.title)
                .font(.headline)
                .lineLimit(1)

            Spacer()

            Picker("View", selection: $viewMode) {
                Text("Splat").tag(ViewMode.splat)
                Text("Mesh").tag(ViewMode.mesh)
            }
            .pickerStyle(.segmented)
            .fixedSize()
            .labelsHidden()

            Spacer()

            Button {
                export()
            } label: {
                Label(viewMode == .splat ? "Export Splat" : "Export Mesh",
                      systemImage: "square.and.arrow.up")
            }
            .disabled(opened.url == nil)
        }
        .padding(12)
        .background(.ultraThinMaterial)
    }

    /// Export follows the current view: the splat PLY in splat mode, the mesh GLB
    /// in mesh mode.
    private func export() {
        switch viewMode {
        case .splat:
            model.exportOpened()
        case .mesh:
            model.exportMesh(method: settings.method,
                             smoothGrid: settings.smoothGrid,
                             depthRatioCull: Float(settings.depthRatioCull),
                             surfelExtent: Float(settings.surfelExtent),
                             poissonResolution: Int(settings.poissonResolution))
        }
    }

    private var hint: some View {
        Text(viewMode == .splat
             ? "Drag to look · WASD to move · Q/E up-down · scroll to dolly · R to reset"
             : "Drag to orbit · scroll to zoom")
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(8)
            .background(.ultraThinMaterial, in: Capsule())
            .padding(.bottom, 12)
    }
}
