import SwiftUI

/// A slim bar docked at the bottom of the window, tracking
/// `GalleryModel.sceneProgress` regardless of what's on screen. Multi-image
/// reconstruction takes minutes (COLMAP + OpenSplat training), so it keeps
/// running — and this keeps updating — across navigation instead of trapping
/// the user on a single blocking screen for the whole wait.
struct SceneProgressBar: View {
    @ObservedObject var model: GalleryModel

    var body: some View {
        if let progress = model.sceneProgress {
            content(for: progress)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(.regularMaterial)
                .overlay(alignment: .top) { Divider() }
                .contentShape(Rectangle())
                .onTapGesture { model.reopenInProgressScene() }
                .transition(.move(edge: .bottom).combined(with: .opacity))
                // Auto-dismiss a finished scene a few seconds after it lands,
                // in case the user wandered off and never taps it.
                .task(id: progress.isComplete ? progress.key : nil) {
                    guard progress.isComplete else { return }
                    try? await Task.sleep(for: .seconds(6))
                    if model.sceneProgress?.key == progress.key { model.dismissSceneProgress() }
                }
        }
    }

    @ViewBuilder
    private func content(for progress: GalleryModel.SceneProgress) -> some View {
        HStack(spacing: 12) {
            Image(systemName: progress.isComplete ? "checkmark.circle.fill" : "cube.transparent")
                .foregroundStyle(progress.isComplete ? .green : .secondary)
                .imageScale(.medium)

            VStack(alignment: .leading, spacing: 2) {
                Text(progress.title).font(.subheadline.weight(.medium))
                Text(progress.isComplete ? "Ready — tap to view" : progress.stageLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(minWidth: 160, alignment: .leading)

            if progress.isComplete {
                Spacer(minLength: 0)
            } else if progress.indeterminate {
                ProgressView().progressViewStyle(.linear)   // no measurable fraction
            } else {
                ProgressView(value: progress.fraction)
                Text("\(Int(progress.fraction * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 36, alignment: .trailing)
            }

            if progress.isComplete {
                Button {
                    model.dismissSceneProgress()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            } else if progress.cancellable {
                Button("Cancel") {
                    model.cancelSceneReconstruction()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }
}
