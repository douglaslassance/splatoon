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
            methodCard(selection: $settings.singleImageMethod, cases: MeshMethod.allCases)
            optionsCard(for: settings.singleImageMethod)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Mesh method", selection: selection) {
                    ForEach(cases) { method in Text(method.displayName).tag(method) }
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()

                Text(selection.wrappedValue.detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        } label: {
            Text("Mesh method").font(.headline)
        }
    }

    @ViewBuilder
    private func optionsCard(for method: MeshMethod) -> some View {
        switch method {
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
