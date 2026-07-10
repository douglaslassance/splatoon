import SwiftUI
import Photos

struct ContentView: View {
    @StateObject private var model = GalleryModel()

    var body: some View {
        VStack(spacing: 0) {
            content
            SceneProgressBar(model: model)
        }
        .animation(.easeInOut(duration: 0.2), value: model.sceneProgress)
        .frame(minWidth: 860, minHeight: 600)
        .onAppear { model.onAppear() }
        .alert("Something went wrong",
               isPresented: Binding(get: { model.errorMessage != nil },
                                    set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
        .alert("Couldn't build the scene",
               isPresented: Binding(get: { model.sceneFailure != nil },
                                    set: { if !$0 { model.sceneFailure = nil } })) {
            Button("Use Single Image") { model.fallbackToSingleImage() }
            Button("Cancel", role: .cancel) { model.sceneFailure = nil }
        } message: {
            Text((model.sceneFailure?.message ?? "") + "\n\nUse single-image reconstruction on the photo you tapped instead?")
        }
    }

    @ViewBuilder
    private var content: some View {
        switch model.authorization {
        case .authorized, .limited:
            if let opened = model.opened {
                SplatDetailView(model: model, opened: opened)
            } else {
                GalleryView(model: model)
            }
        case .notDetermined:
            AccessGate(
                systemImage: "photo.on.rectangle.angled",
                title: "Access Your Photos",
                message: "Splatoon turns your photos into 3D Gaussian splats, on-device.",
                buttonTitle: "Grant Access",
                action: { model.requestAccess() }
            )
        default:
            AccessGate(
                systemImage: "lock",
                title: "Photo Access Needed",
                message: "Enable photo access in System Settings › Privacy & Security › Photos.",
                buttonTitle: "Open Settings",
                action: {
                    if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Photos") {
                        NSWorkspace.shared.open(url)
                    }
                }
            )
        }
    }
}

private struct AccessGate: View {
    let systemImage: String
    let title: String
    let message: String
    let buttonTitle: String
    let action: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: systemImage)
                .font(.system(size: 56))
                .foregroundStyle(.secondary)
            Text(title).font(.largeTitle.bold())
            Text(message)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 360)
            Button(buttonTitle, action: action)
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

