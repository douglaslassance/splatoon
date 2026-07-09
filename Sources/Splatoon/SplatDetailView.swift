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
            .overlay { if let info = progressInfo { progressOverlay(info) } }
            .overlay(alignment: .top) { toolbar }
            .overlay(alignment: .bottom) { if progressInfo == nil && !isGenerating { hint } }
            .animation(.easeInOut(duration: 0.2), value: progressInfo)
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
            SplatViewer(url: opened.url, initialPose: opened.initialPose, onLoadingChange: { isLoading = $0 })
        case .mesh:
            if let mesh = model.openedMesh {
                MeshViewer(mesh: mesh, initialPose: opened.initialPose)
            } else {
                Color.clear   // progress overlay covers the build
            }
        }
    }

    /// Identity for the mesh-rebuild task: mode, splat, availability, and settings.
    private var meshTaskID: String {
        "\(viewMode.rawValue)|\(opened.id)|\(opened.url != nil)|\(settings.signature)"
    }

    private struct ProgressInfo: Equatable {
        var text: String
        var fraction: Double?   // nil = indeterminate
    }

    /// Whether the opened splat (single- or multi-image) is still generating —
    /// the docked bar (visible from anywhere, not just this screen) already shows
    /// its staged progress, so the center overlay stays out of the way instead of
    /// duplicating it.
    private var isGenerating: Bool {
        opened.url == nil && model.sceneProgress?.key == opened.id
    }

    /// The progress to show over the view, or nil when it's interactive (or
    /// already shown in the docked bar — see `isGenerating`).
    private var progressInfo: ProgressInfo? {
        if isGenerating { return nil }
        if let busy = model.busyMessage { return ProgressInfo(text: busy, fraction: nil) }
        if opened.url == nil { return ProgressInfo(text: "Generating splat…", fraction: nil) }
        switch viewMode {
        case .splat:
            return isLoading ? ProgressInfo(text: "Loading splat…", fraction: nil) : nil
        case .mesh:
            return model.openedMesh == nil ? ProgressInfo(text: "Building mesh…", fraction: nil) : nil
        }
    }

    private func progressOverlay(_ info: ProgressInfo) -> some View {
        ZStack {
            Color.black.opacity(0.35).ignoresSafeArea()
            VStack(spacing: 14) {
                if let fraction = info.fraction {
                    ProgressView(value: fraction).frame(width: 220)
                } else {
                    ProgressView().controlSize(.large)
                }
                Text(info.text).foregroundStyle(.white)
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
                model.openInSuperSplat()
            } label: {
                Label("Clean Up", systemImage: "wand.and.stars")
            }
            .help("Open this splat in SuperSplat (free browser editor) to remove floaters, crop, and publish. "
                  + "The file is revealed in Finder — drag it into the editor tab.")
            .disabled(opened.url == nil)

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
        Text("Drag to look · Right-drag to pan · WASD to move · Q/E up-down · Scroll to dolly · R to reset")
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(8)
            .background(.ultraThinMaterial, in: Capsule())
            .padding(.bottom, 12)
    }
}
