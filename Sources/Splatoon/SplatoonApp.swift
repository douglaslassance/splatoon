import SwiftUI
import Metal
import MetalSplatter

@main
struct SplatoonApp: App {
    init() {
        if CommandLine.arguments.contains("--selftest-renderer") {
            runRendererSelfTest()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 860, minHeight: 600)
        }
        .windowResizability(.contentMinSize)
    }
}

/// Creates a `SplatRenderer` in the real app-bundle context to verify that
/// MetalSplatter's `Bundle.module` resolves and its `default.metallib` loads.
/// Run headlessly: `Splatoon.app/Contents/MacOS/Splatoon --selftest-renderer`.
private func runRendererSelfTest() -> Never {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("SELFTEST: no Metal device")
        exit(1)
    }
    do {
        _ = try SplatRenderer(device: device,
                              colorFormat: .bgra8Unorm_srgb,
                              depthFormat: .depth32Float,
                              sampleCount: 1,
                              maxViewCount: 1,
                              maxSimultaneousRenders: 3)
        print("SELFTEST: SplatRenderer created OK — metallib resolved via Bundle.module")
        exit(0)
    } catch {
        print("SELFTEST: SplatRenderer threw: \(error)")
        exit(1)
    }
}
