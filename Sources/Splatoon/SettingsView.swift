import SwiftUI

/// The app Settings panel (⌘,), split into tabs so each fits without scrolling:
/// single-image (SHARP) meshing, and multi-input (scene) reconstruction + meshing.
struct SettingsView: View {
    @ObservedObject var settings: MeshSettings

    var body: some View {
        TabView {
            singleImageTab
                .tabItem { Label("Single Image", systemImage: "photo") }
            multiImageTab
                .tabItem { Label("Multi-Input", systemImage: "square.stack.3d.up") }
            CacheTab()
                .tabItem { Label("Cache", systemImage: "internaldrive") }
        }
        .frame(width: 480)
        .padding(20)
    }

    // MARK: - Tabs

    private var singleImageTab: some View {
        VStack(alignment: .leading, spacing: 16) {
            generatorCard
            methodCard(selection: $settings.singleImageMethod, cases: MeshMethod.singleImageCases)
            optionsCard(for: settings.singleImageMethod)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Chooses how a single photo becomes a splat: SHARP (fast relief) or
    /// TripoSplat (full 3D object). The TripoSplat row greys out when `tripo-cli`
    /// isn't installed — the same treatment as an unavailable mesh method.
    private var generatorCard: some View {
        radioCard("Splat method", items: SingleImageGenerator.allCases,
                  selection: $settings.singleImageGenerator,
                  label: { $0.displayName }, detail: { $0.detail },
                  unavailable: Self.generatorUnavailableHint) {
            if settings.singleImageGenerator == .triposplat {
                sliderRow(title: "Gaussians",
                          valueText: "\(Int(settings.triposplatGaussians))",
                          value: $settings.triposplatGaussians, range: 32768...262144, step: 32768,
                          caption: "How many Gaussians TripoSplat generates. More gives finer detail and a "
                            + "larger file, at a little more time.")
            }
        }
    }

    private var multiImageTab: some View {
        VStack(alignment: .leading, spacing: 16) {
            sceneDetectionCard
            methodCard(selection: $settings.sceneMethod, cases: MeshMethod.sceneCases)
            optionsCard(for: settings.sceneMethod)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Cards

    private var sceneDetectionCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Toggle("Reconstruct multi-input scenes", isOn: $settings.useMultiImageReconstruction)
                    Text("When several photos or videos capture the same place (or you open one video), "
                         + "reconstruct a multi-view splat (COLMAP + OpenSplat) instead of a single-image one. "
                         + "Blurry and badly-exposed frames are dropped automatically.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Picker("Group by", selection: $settings.sceneMatchMode) {
                        ForEach(SceneGrouping.MatchMode.allCases) { mode in
                            Text(mode.displayName).tag(mode)
                        }
                    }
                    .pickerStyle(.menu)
                    .fixedSize()
                    Text("“Time and location” groups shots taken close together in the same place. "
                         + "“Location only” groups everything shot at that place, even on different days "
                         + "(needs location data on the photos).")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .disabled(!settings.useMultiImageReconstruction)
                .opacity(settings.useMultiImageReconstruction ? 1 : 0.5)

                sliderRow(title: "Training steps",
                          valueText: "\(Int(settings.multiImageIterations)) (~\(estimatedMinutes) min)",
                          value: $settings.multiImageIterations, range: 1000...30000, step: 500,
                          caption: "How long the multi-view trainer runs. Too few and the splat stays blurry. "
                              + "Quality keeps improving well past 15000. Higher is sharper but slower. "
                              + "Time is a rough estimate (~13 steps/sec) — actual speed depends on your Mac.")
                    .disabled(!settings.useMultiImageReconstruction)
                    .opacity(settings.useMultiImageReconstruction ? 1 : 0.5)

                VStack(alignment: .leading, spacing: 4) {
                    Picker("Trainer", selection: $settings.sceneTrainer) {
                        ForEach(SplatTrainer.allCases) { trainer in
                            Text(trainer.displayName).tag(trainer)
                        }
                    }
                    .pickerStyle(.menu)
                    .fixedSize()
                    Text("OpenSplat is the reliable default. Brush is a native-Metal trainer (no libtorch), "
                         + "usually much faster on Apple GPUs. If Brush isn't installed, OpenSplat is used instead.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .disabled(!settings.useMultiImageReconstruction)
                .opacity(settings.useMultiImageReconstruction ? 1 : 0.5)

                VStack(alignment: .leading, spacing: 4) {
                    Picker("Colour detail", selection: $settings.sceneSHDegree) {
                        Text("Standard").tag(0)
                        Text("Full").tag(3)
                    }
                    .pickerStyle(.segmented)
                    Text("Standard stores one flat colour per splat, the most compact option and plenty "
                         + "for most captures. Full adds view-dependent shine at several times the file "
                         + "size and slower rendering.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .disabled(!settings.useMultiImageReconstruction)
                .opacity(settings.useMultiImageReconstruction ? 1 : 0.5)

                VStack(alignment: .leading, spacing: 4) {
                    Toggle("Global pose solver", isOn: $settings.useGlobalPoseSolver)
                    Text("Solve camera poses with COLMAP's global method (global_mapper) instead of its "
                         + "incremental one. More robust on sparse or weakly-overlapping captures, where the "
                         + "incremental solver often aligns only a fraction of the shots. Falls back "
                         + "automatically on older COLMAP builds.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .disabled(!settings.useMultiImageReconstruction)
                .opacity(settings.useMultiImageReconstruction ? 1 : 0.5)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        } label: {
            Text("Scene detection").font(.headline)
        }
    }

    private func methodCard(selection: Binding<MeshMethod>, cases: [MeshMethod]) -> some View {
        radioCard("Mesh method", items: cases, selection: selection,
                  label: { $0.displayName }, detail: { $0.detail },
                  unavailable: Self.meshUnavailableHint) { EmptyView() }
    }

    /// A radio-button card whose individual items can be greyed out (disabled) with
    /// a shared "how to enable" hint — the single, consistent way the app shows an
    /// option that needs a tool that isn't installed (Photogrammetry, TripoSplat).
    /// `extra` holds any controls specific to the selected item.
    @ViewBuilder
    private func radioCard<T: Hashable & Identifiable, Extra: View>(
        _ title: String, items: [T], selection: Binding<T>,
        label: @escaping (T) -> String, detail: @escaping (T) -> String,
        unavailable: @escaping (T) -> String?,
        @ViewBuilder extra: () -> Extra
    ) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(items) { item in
                    let disabled = unavailable(item) != nil
                    Button { selection.wrappedValue = item } label: {
                        HStack(spacing: 6) {
                            Image(systemName: selection.wrappedValue == item ? "largecircle.fill.circle" : "circle")
                            Text(label(item))
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(disabled)
                    .foregroundStyle(disabled ? AnyShapeStyle(.tertiary) : AnyShapeStyle(.primary))
                }

                Text(detail(selection.wrappedValue))
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                // Show the hint for whichever offered item is unavailable.
                if let hint = items.compactMap(unavailable).first {
                    Label(hint, systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }

                extra()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        } label: {
            Text(title).font(.headline)
        }
    }

    /// Install-hint for a mesh method that needs an absent tool, else nil.
    private static func meshUnavailableHint(_ method: MeshMethod) -> String? {
        if method == .photogrammetry && !ToolLocator.photogrammetryAvailable {
            return "OpenMVS isn't installed. Run scripts/fetch-tools.sh to enable photogrammetry."
        }
        return nil
    }

    /// Install-hint for a splat generator that needs an absent tool, else nil.
    private static func generatorUnavailableHint(_ gen: SingleImageGenerator) -> String? {
        if gen == .triposplat && !ToolLocator.tripoSplatAvailable {
            return "TripoSplat needs the tripo-cli tool: install uv, then "
                + "`uv tool install tripo-cli` and `tripo-cli download`."
        }
        return nil
    }

    @ViewBuilder
    private func optionsCard(for method: MeshMethod) -> some View {
        switch method {
        case .photogrammetry:
            GroupBox {
                VStack(alignment: .leading, spacing: 14) {
                    sliderRow(title: "Detail",
                              valueText: String(format: "%.0f%%", settings.photogrammetryQuality * 100),
                              value: $settings.photogrammetryQuality, range: 0...1,
                              caption: "How finely the dense stereo reconstructs. Higher resolves more detail from the photos but is markedly slower and heavier.")
                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("Refine mesh", isOn: $settings.photogrammetryRefine)
                        Text("Runs OpenMVS's extra photometric refinement pass for crisper geometry. Noticeably slower.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
            } label: {
                Text("Photogrammetry options").font(.headline)
            }
        case .fusion:
            GroupBox {
                VStack(alignment: .leading, spacing: 14) {
                    sliderRow(title: "Resolution",
                              valueText: "\(Int(settings.poissonResolution))",
                              value: $settings.poissonResolution, range: 128...4096, step: 128,
                              caption: "Sets the voxel grid resolution the fused depth is meshed at. Higher means finer detail, at the cost of speed and file size.")
                    sliderRow(title: "Max views",
                              valueText: "\(Int(settings.fusionMaxViews))",
                              value: $settings.fusionMaxViews, range: 4...200, step: 4,
                              caption: "How many registered cameras to render and fuse, sampled evenly across the capture. More views give fuller coverage and less noise, but take proportionally longer.")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
            } label: {
                Text("Fusion options").font(.headline)
            }
        case .density:
            GroupBox {
                VStack(alignment: .leading, spacing: 14) {
                    sliderRow(title: "Resolution",
                              valueText: "\(Int(settings.poissonResolution))",
                              value: $settings.poissonResolution, range: 128...4096, step: 128,
                              caption: "Sets the voxel grid resolution. The higher the value, the finer the detail, at the cost of speed and file size. It is capped by the splat's own point density.")
                    sliderRow(title: "Surface tightness",
                              valueText: String(format: "%.2f", settings.surfaceTightness),
                              value: $settings.surfaceTightness, range: 0...1,
                              caption: "How tightly the surface hugs the splats. Higher pulls it toward the densest cores (thinner, more detail but more holes). Lower gives a fuller, smoother shell.")
                    sliderRow(title: "Offset",
                              valueText: String(format: "%+.2f", settings.densityOffset),
                              value: $settings.densityOffset, range: -1...1,
                              caption: "Shifts the surface outward (positive, inflate) or inward (negative). Useful to close small gaps or trim overshoot.")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
            } label: {
                Text("Density options").font(.headline)
            }
        case .grid:
            GroupBox {
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("Smooth surface", isOn: $settings.smoothGrid)
                        Text("Smooths depth noise across the surface. Fine detail gets rounded off when enabled.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    sliderRow(title: "Occlusion cull",
                              valueText: String(format: "%.2f", settings.depthRatioCull),
                              value: $settings.depthRatioCull, range: 1.05...3.0,
                              caption: "Cuts triangles that straddle a depth discontinuity. The lower the value, the more it cuts, leaving more holes at occlusion edges.")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
            } label: {
                Text("Grid options").font(.headline)
            }
        case .surfel:
            GroupBox {
                sliderRow(title: "Quad size",
                          valueText: String(format: "%.1f×", settings.surfelExtent),
                          value: $settings.surfelExtent, range: 0.5...4.0,
                          caption: "Sizes each quad as a multiple of its splat's radius. The lower the value, the sharper and finer the surface, though gaps may appear between quads.")
                    .padding(6)
            } label: {
                Text("Surfel options").font(.headline)
            }
        case .poisson:
            GroupBox {
                sliderRow(title: "Resolution",
                          valueText: "\(Int(settings.poissonResolution))",
                          value: $settings.poissonResolution, range: 128...4096, step: 128,
                          caption: "Sets the voxel grid resolution. The higher the value, the finer the detail, at the cost of speed and file size. It is capped by the splat's own point density.")
                    .padding(6)
            } label: {
                Text("Poisson options").font(.headline)
            }
        }
    }

    private var estimatedMinutes: Int {
        max(1, Int((settings.multiImageIterations / 12.85 / 60).rounded()))
    }

    /// A label + value header on one line, with a full-width slider beneath it.
    @ViewBuilder
    private func sliderRow(title: String, valueText: String,
                           value: Binding<Double>, range: ClosedRange<Double>,
                           step: Double? = nil, caption: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                Spacer()
                Text(valueText).foregroundStyle(.secondary).monospacedDigit()
            }
            if let step {
                Slider(value: value, in: range, step: step)
            } else {
                Slider(value: value, in: range)
            }
            Text(caption)
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The Cache tab: shows how much disk the generated splats occupy, and empties
/// it on demand. Sizing runs off the main thread so a large cache doesn't hitch
/// the panel.
private struct CacheTab: View {
    @State private var sizeBytes: Int64 = 0
    @State private var isWorking = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            GroupBox {
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        Text("Cached splats")
                        Spacer()
                        Text(sizeText).foregroundStyle(.secondary).monospacedDigit()
                    }
                    Text("Generated splats (single-image and multi-input scenes), their camera "
                         + "poses, and mesh previews are cached on disk so reopening is instant. "
                         + "Clearing frees the space. Everything regenerates on demand.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 8) {
                        Button("Clear Cache", role: .destructive, action: clear)
                            .disabled(sizeBytes == 0 || isWorking)
                        if isWorking { ProgressView().controlSize(.small) }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
            } label: {
                Text("Storage").font(.headline)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onAppear(perform: refresh)
    }

    private var sizeText: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }

    private func refresh() {
        DispatchQueue.global(qos: .userInitiated).async {
            let s = SplatCache.size()
            DispatchQueue.main.async { sizeBytes = s }
        }
    }

    private func clear() {
        isWorking = true
        DispatchQueue.global(qos: .userInitiated).async {
            SplatCache.clear()
            let s = SplatCache.size()
            DispatchQueue.main.async {
                sizeBytes = s
                isWorking = false
            }
        }
    }
}
