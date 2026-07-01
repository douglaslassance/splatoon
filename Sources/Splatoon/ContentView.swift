import SwiftUI
import Photos

struct ContentView: View {
    @StateObject private var model = GalleryModel()

    var body: some View {
        ZStack {
            content
            if let message = model.busyMessage {
                BusyOverlay(message: message)
            }
        }
        .frame(minWidth: 860, minHeight: 600)
        .onAppear { model.onAppear() }
        .alert("Something went wrong",
               isPresented: Binding(get: { model.errorMessage != nil },
                                    set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
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

private struct BusyOverlay: View {
    let message: String

    var body: some View {
        ZStack {
            Color.black.opacity(0.4).ignoresSafeArea()
            VStack(spacing: 14) {
                ProgressView().controlSize(.large)
                Text(message).foregroundStyle(.white)
            }
            .padding(28)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
    }
}
