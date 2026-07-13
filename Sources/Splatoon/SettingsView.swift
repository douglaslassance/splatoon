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
                          caption: "How long the multi-view trainer runs. Too few and the splat stays blurry; "
                              + "quality keeps improving well past 15000. Higher is sharper but slower. "
                              + "Time is a rough estimate (~13 steps/sec) — actual speed depends on your Mac.")
                    .disabled(!settings.useMultiImageReconstruction)
                    .opacity(settings.useMultiImageReconstruction ? 1 : 0.5)

                VStack(alignment: .leading, spacing: 4) {
                    Picker("Colour detail", selection: $settings.sceneSHDegree) {
                        Text("Standard").tag(1)
                        Text("High").tag(2)
                        Text("Maximum").tag(3)
                    }
                    .pickerStyle(.segmented)
                    Text("Spherical-harmonics degree. Higher captures more view-dependent shine but "
                         + "multiplies the splat's colour data (Maximum files are several times larger and "
                         + "slower to render). Standard is plenty for most captures.")
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

                Label {
                    Text("For best results use a **video** (or several), or many overlapping photos in a slow "
                         + "continuous orbit. Spread-out angles won't align.")
                        .font(.caption).foregroundStyle(.secondary)
                } icon: {
                    Image(systemName: "video").foregroundStyle(.secondary)
                }
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
                              caption: "How tightly the surface hugs the splats. Higher pulls it toward the densest cores (thinner, more detail but more holes); lower gives a fuller, smoother shell.")
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
