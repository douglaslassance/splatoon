import SwiftUI

/// Full-window viewer shown after opening an image. Toggles between the Gaussian
/// splat and its triangle-mesh preview, with a back button and a context-aware
/// export that follows whichever view is showing.
struct SplatDetailView: View {
    @ObservedObject var model: GalleryModel
    @EnvironmentObject var settings: MeshSettings
    let opened: GalleryModel.OpenedSplat
    @State private var viewMode: ViewMode = .splat

    enum ViewMode: String { case splat, mesh }

    var body: some View {
        content
            .ignoresSafeArea()
            .overlay(alignment: .top) { toolbar }
            // All progress (generation, load, mesh build) shows in the docked bar
            // at the window bottom; the hint hides while this splat has progress.
            .overlay(alignment: .bottom) { if !hasProgress { hint } }
            // Rebuild the mesh preview when entering mesh mode, changing the opened
            // splat, or tweaking mesh settings. `.task(id:)` cancels/reruns on change.
            .task(id: meshTaskID) {
                guard viewMode == .mesh, opened.url != nil else { return }
                model.buildOpenedMesh(settingsSignature: settings.signature(forScene: opened.isScene),
                                      method: settings.method(forScene: opened.isScene),
                                      smoothGrid: settings.smoothGrid,
                                      depthRatioCull: Float(settings.depthRatioCull),
                                      surfelExtent: Float(settings.surfelExtent),
                                      poissonResolution: Int(settings.poissonResolution),
                                      surfaceTightness: Float(settings.surfaceTightness),
                                      densityOffset: Float(settings.densityOffset))
            }
    }

    @ViewBuilder
    private var content: some View {
        switch viewMode {
        case .splat:
            SplatViewer(url: opened.url, initialPose: opened.initialPose,
                        onLoadingChange: { model.setSplatLoading($0) })
        case .mesh:
            if let mesh = model.openedMesh {
                MeshViewer(mesh: mesh, initialPose: opened.initialPose)
            } else {
                Color.clear   // the docked bar shows "Building mesh…"
            }
        }
    }

    /// Identity for the mesh-rebuild task: mode, splat, availability, and settings.
    private var meshTaskID: String {
        "\(viewMode.rawValue)|\(opened.id)|\(opened.url != nil)|\(settings.signature(forScene: opened.isScene))"
    }

    /// Whether the docked bar is showing progress for this splat (generation,
    /// load, or mesh build) — used to hide the control hint meanwhile.
    private var hasProgress: Bool {
        opened.url == nil || model.sceneProgress?.key == opened.id
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
                model.regenerateOpened(iterations: Int(settings.multiImageIterations),
                                       matchMode: settings.sceneMatchMode)
            } label: {
                Label("Regenerate", systemImage: "arrow.clockwise")
            }
            .labelStyle(.iconOnly)
            .help("Discard the cached splat and rebuild it from the original photo(s) or video.")
            .disabled(opened.url == nil)

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
            model.exportMesh(method: settings.method(forScene: opened.isScene),
                             smoothGrid: settings.smoothGrid,
                             depthRatioCull: Float(settings.depthRatioCull),
                             surfelExtent: Float(settings.surfelExtent),
                             poissonResolution: Int(settings.poissonResolution),
                             surfaceTightness: Float(settings.surfaceTightness),
                             densityOffset: Float(settings.densityOffset))
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
