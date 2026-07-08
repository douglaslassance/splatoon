import SwiftUI

/// The app Settings panel (⌘,). Lets the user choose the mesh export method and
/// tune its parameters. Uses GroupBox cards so sliders span the full width.
struct SettingsView: View {
    @ObservedObject var settings: MeshSettings

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                sceneDetectionCard
                methodCard
                optionsCard
            }
            .padding(20)
        }
        .frame(width: 460)
        .frame(minHeight: 360)
    }

    private var sceneDetectionCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 4) {
                Toggle("Reconstruct multi-image scenes", isOn: $settings.useMultiImageReconstruction)
                Text("When a photo has several same-place/time siblings, combine them into one "
                     + "multi-view splat (COLMAP + OpenSplat) instead of using just the tapped photo. "
                     + "Off always uses single-image reconstruction.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        } label: {
            Text("Scene detection").font(.headline)
        }
    }

    private var methodCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Method", selection: $settings.method) {
                    ForEach(MeshMethod.allCases) { method in
                        Text(method.displayName).tag(method)
                    }
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()

                Text(settings.method.detail)
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
    private var optionsCard: some View {
        switch settings.method {
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
