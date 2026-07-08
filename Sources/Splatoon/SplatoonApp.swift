import SwiftUI
import Metal
import MetalSplatter

@main
struct SplatoonApp: App {
    @StateObject private var meshSettings = MeshSettings()

    init() {
        if CommandLine.arguments.contains("--selftest-renderer") {
            runRendererSelfTest()
        }
        if CommandLine.arguments.contains("--selftest-mesh") {
            runMeshSelfTest()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(meshSettings)
                .frame(minWidth: 860, minHeight: 600)
        }
        .windowResizability(.contentMinSize)

        Settings {
            SettingsView(settings: meshSettings)
        }
    }
}

/// Builds a mesh from a cached splat PLY headlessly, for verification.
/// Usage: `Splatoon --selftest-mesh <in.ply> <out.glb> [method] [resolution]`.
private func runMeshSelfTest() -> Never {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: "--selftest-mesh"), args.count > idx + 2 else {
        print("SELFTEST mesh: usage --selftest-mesh <in.ply> <out.glb> [method] [resolution]")
        exit(1)
    }
    let ply = URL(fileURLWithPath: args[idx + 1])
    let out = URL(fileURLWithPath: args[idx + 2])
    let method = (args.count > idx + 3 ? MeshMethod(rawValue: args[idx + 3]) : nil) ?? .grid
    let resolution = (args.count > idx + 4 ? Int(args[idx + 4]) : nil) ?? 256
    do {
        let gaussians = try SplatPLYReader.readGaussians(from: ply)
        try MeshExporter.saveGLB(gaussians: gaussians, to: out, method: method,
                                 poissonResolution: resolution)
        print("SELFTEST mesh OK [\(method.rawValue)] -> \(out.path)")
        exit(0)
    } catch {
        print("SELFTEST mesh failed: \(error)")
        exit(1)
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
