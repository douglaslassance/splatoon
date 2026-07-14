import Foundation

// MARK: - Photogrammetry mesh (COLMAP + OpenMVS)
//
// Turns a scene's registered images into a watertight, UV-textured mesh via the
// classic dense-MVS pipeline — the same technique standalone photogrammetry tools
// use, and a large step up from meshing a Gaussian splat (which is not
// surface-aligned). It reuses the images + COLMAP model persisted next to the
// splat (`MultiImageReconstructor.meshSourceURL`) and shells out to COLMAP
// (`image_undistorter`) then OpenMVS:
//
//   image_undistorter -> InterfaceCOLMAP -> DensifyPointCloud
//                     -> ReconstructMesh -> [RefineMesh] -> TextureMesh
//
// The output is a textured OBJ bundle (model.obj + .mtl + texture images) written
// into the splat's `.mesh` cache bundle, displayed by `MeshViewer`'s OBJ path and
// exported as-is. Blocking; drive it from a background task. Cancellable.

final class PhotogrammetryMesher {

    enum MeshError: LocalizedError {
        case toolFailed(tool: String, stage: String, code: Int32, log: String)
        case noMeshSource
        case noOutput
        case cancelled

        var errorDescription: String? {
            switch self {
            case .toolFailed(let tool, let stage, let code, let log):
                return "\(tool) failed during \(stage) (exit \(code)).\n\(log)"
            case .noMeshSource:
                return "This scene has no saved photogrammetry source. Regenerate it "
                    + "(scenes made before photogrammetry was added don't keep their images)."
            case .noOutput:
                return "The photogrammetry pipeline produced no mesh."
            case .cancelled:
                return "Mesh generation cancelled."
            }
        }
    }

    private let colmap: URL
    private let interfaceCOLMAP: URL
    private let densifyPointCloud: URL
    private let reconstructMesh: URL
    private let textureMesh: URL
    private let refineMeshTool: URL?

    /// DensifyPointCloud resolution level: 0 = full image resolution (slowest,
    /// most detail), higher downscales. Driven by the "Quality" setting.
    var resolutionLevel = 1
    /// Run the (slow) RefineMesh stage for extra geometric detail.
    var refineMesh = false

    private let lock = NSLock()
    private var currentProcess: Process?
    private var cancelled = false

    /// All OpenMVS binaries must resolve; `image_undistorter` comes from COLMAP.
    init?() {
        guard let colmap = ToolLocator.resolvedURL(for: .colmap),
              let iface = ToolLocator.resolvedURL(for: .interfaceCOLMAP),
              let densify = ToolLocator.resolvedURL(for: .densifyPointCloud),
              let recon = ToolLocator.resolvedURL(for: .reconstructMesh),
              let texture = ToolLocator.resolvedURL(for: .textureMesh) else {
            return nil
        }
        self.colmap = colmap
        self.interfaceCOLMAP = iface
        self.densifyPointCloud = densify
        self.reconstructMesh = recon
        self.textureMesh = texture
        self.refineMeshTool = ToolLocator.resolvedURL(for: .refineMesh)
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let process = currentProcess
        lock.unlock()
        process?.terminate()
    }

    private var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelled
    }

    /// The textured-mesh bundle directory for a splat + quality settings:
    /// `<stem>-r<level>[-refine].mesh`, holding `model.obj` + `.mtl` + textures.
    /// The settings are in the name so changing Detail/Refine builds (and caches) a
    /// distinct mesh rather than silently reusing a stale one.
    static func meshBundleURL(for plyURL: URL, resolutionLevel: Int, refine: Bool) -> URL {
        let base = plyURL.deletingPathExtension().lastPathComponent
            + "-r\(resolutionLevel)" + (refine ? "-refine" : "")
        return plyURL.deletingLastPathComponent().appendingPathComponent(base + ".mesh")
    }

    /// The OBJ inside a splat's mesh bundle (what the viewer/export load).
    static func meshOBJURL(for plyURL: URL, resolutionLevel: Int, refine: Bool) -> URL {
        meshBundleURL(for: plyURL, resolutionLevel: resolutionLevel, refine: refine)
            .appendingPathComponent("model.obj")
    }

    /// Run the full pipeline. `meshSource` is the persisted bundle (`images/` +
    /// `sparse/0`); `workDir` is scratch; the textured OBJ is written into
    /// `bundle` (`<stem>.mesh`). Returns the OBJ URL. `progress` reports a stage
    /// label + overall 0…1 fraction.
    @discardableResult
    func run(meshSource: URL, workDir: URL, bundle: URL,
             progress: @escaping (ReconstructionProgress) -> Void) throws -> URL {
        let fm = FileManager.default
        let images = meshSource.appendingPathComponent("images")
        let sparse0 = meshSource.appendingPathComponent("sparse/0")
        guard fm.fileExists(atPath: images.path), fm.fileExists(atPath: sparse0.path) else {
            throw MeshError.noMeshSource
        }

        try? fm.removeItem(at: workDir)
        try fm.createDirectory(at: workDir, withIntermediateDirectories: true)
        try? fm.removeItem(at: bundle)
        try fm.createDirectory(at: bundle, withIntermediateDirectories: true)

        // Refining shifts the band boundaries so every stage still adds up to 1.
        let reconEnd = refineMesh ? 0.72 : 0.80
        var bands: [(stage: String, start: Double, end: Double)] = [
            ("Undistorting images…", 0.00, 0.08),
            ("Preparing scene…", 0.08, 0.12),
            ("Densifying point cloud…", 0.12, 0.55),
            ("Reconstructing mesh…", 0.55, reconEnd),
        ]
        if refineMesh { bands.append(("Refining mesh…", reconEnd, 0.85)) }
        bands.append(("Texturing mesh…", refineMesh ? 0.85 : reconEnd, 1.00))

        func report(_ band: (stage: String, start: Double, end: Double), within: Double) {
            let f = band.start + (band.end - band.start) * max(0, min(1, within))
            progress(ReconstructionProgress(stageLabel: band.stage, fraction: f))
        }
        /// A progress sink that parses any "NN%" the tool prints.
        func percentSink(_ band: (stage: String, start: Double, end: Double)) -> (String) -> Void {
            { line in
                guard let caps = Self.captures(#"(\d+)%"#, in: line), let p = Double(caps[0]) else { return }
                report(band, within: p / 100)
            }
        }

        // 1. Undistort the registered images into a PINHOLE COLMAP workspace, so
        //    OpenMVS gets distortion-free images + cameras.
        let dense = workDir.appendingPathComponent("dense", isDirectory: true)
        report(bands[0], within: 0)
        try runTool(colmap, stage: "undistortion", workDir: workDir, args: [
            "image_undistorter",
            "--image_path", images.path,
            "--input_path", sparse0.path,
            "--output_path", dense.path,
            "--output_type", "COLMAP",
        ], onLine: percentSink(bands[0]))

        // 2. Import the COLMAP workspace into an OpenMVS scene.
        report(bands[1], within: 0)
        try runTool(interfaceCOLMAP, stage: "interface", workDir: workDir, args: [
            "-i", dense.path,
            "-o", workDir.appendingPathComponent("scene.mvs").path,
            "--image-folder", dense.appendingPathComponent("images").path,
        ], onLine: percentSink(bands[1]))

        // 3. Dense point cloud (CPU: --cuda-device -1). The slow stage.
        report(bands[2], within: 0)
        try runTool(densifyPointCloud, stage: "densify", workDir: workDir, args: [
            "scene.mvs",
            "-o", "scene_dense.mvs",
            "--resolution-level", String(resolutionLevel),
            "--cuda-device", "-1",
            "-w", workDir.path,
        ], onLine: percentSink(bands[2]))

        // 4. Watertight mesh from the dense cloud.
        report(bands[3], within: 0)
        try runTool(reconstructMesh, stage: "reconstruct", workDir: workDir, args: [
            "scene_dense.mvs",
            "-o", "mesh.ply",
            "-w", workDir.path,
        ], onLine: percentSink(bands[3]))

        // 5. Optional refinement (photometric mesh optimization).
        var meshPly = "mesh.ply"
        if refineMesh, let refineMeshTool {
            let band = bands[4]
            report(band, within: 0)
            try runTool(refineMeshTool, stage: "refine", workDir: workDir, args: [
                "scene_dense.mvs",
                "-m", "mesh.ply",
                "-o", "mesh_refined.ply",
                "-w", workDir.path,
            ], onLine: percentSink(band))
            if fm.fileExists(atPath: workDir.appendingPathComponent("mesh_refined.ply").path) {
                meshPly = "mesh_refined.ply"
            }
        }

        // 6. Texture the mesh, writing the OBJ bundle straight into `bundle` (cwd =
        //    bundle so model.obj/.mtl/textures land together with valid relative
        //    references). Inputs are passed absolute.
        let textureBand = bands.last!
        report(textureBand, within: 0)
        try runTool(textureMesh, stage: "texture", workDir: bundle, args: [
            workDir.appendingPathComponent("scene_dense.mvs").path,
            "-m", workDir.appendingPathComponent(meshPly).path,
            "-o", "model.obj",
            "--export-type", "obj",
            "-w", bundle.path,
        ], onLine: percentSink(textureBand))

        let obj = bundle.appendingPathComponent("model.obj")
        guard fm.fileExists(atPath: obj.path) else { throw MeshError.noOutput }
        // TextureMesh ran with cwd = bundle, so its stage log landed next to the
        // OBJ; drop it so the exported bundle is just mesh + textures.
        try? fm.removeItem(at: bundle.appendingPathComponent("texture.log"))
        try? fm.removeItem(at: workDir)
        report(textureBand, within: 1)
        return obj
    }

    // MARK: - Subprocess (mirrors MultiImageReconstructor.runTool)

    /// Run one tool to completion, streaming stdout+stderr to `<stage>.log` and,
    /// line by line, to `onLine` for live progress. Throws on nonzero exit.
    private func runTool(_ executable: URL, stage: String, workDir: URL, args: [String],
                         onLine: ((String) -> Void)? = nil) throws {
        if isCancelled { throw MeshError.cancelled }

        let fm = FileManager.default
        try? fm.createDirectory(at: workDir, withIntermediateDirectories: true)
        let logURL = workDir.appendingPathComponent("\(stage).log")
        fm.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try FileHandle(forWritingTo: logURL)
        defer { try? logHandle.close() }

        let pipe = Pipe()
        var lineBuffer = Data()
        let drained = DispatchSemaphore(value: 0)
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                pipe.fileHandleForReading.readabilityHandler = nil
                drained.signal()
                return
            }
            logHandle.write(data)
            guard let onLine else { return }
            lineBuffer.append(data)
            // OpenMVS redraws progress with carriage returns, so split on \r and \n.
            while let idx = lineBuffer.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
                let lineData = lineBuffer[lineBuffer.startIndex..<idx]
                lineBuffer.removeSubrange(lineBuffer.startIndex...idx)
                if let line = String(data: lineData, encoding: .utf8) { onLine(line) }
            }
        }

        let process = Process()
        process.executableURL = executable
        process.arguments = args
        process.currentDirectoryURL = workDir
        process.standardOutput = pipe
        process.standardError = pipe

        lock.lock(); currentProcess = process; lock.unlock()

        do {
            try process.run()
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            throw MeshError.toolFailed(tool: executable.lastPathComponent, stage: stage,
                                       code: -1, log: error.localizedDescription)
        }
        process.waitUntilExit()
        drained.wait()

        lock.lock(); currentProcess = nil; lock.unlock()

        if isCancelled { throw MeshError.cancelled }
        if process.terminationStatus != 0 {
            let log = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
            throw MeshError.toolFailed(tool: executable.lastPathComponent, stage: stage,
                                       code: process.terminationStatus, log: String(log.suffix(1500)))
        }
    }

    /// First capture group of `pattern` in `line`, or nil. Raw NSRegularExpression
    /// pattern (Swift regex literals parse ambiguously here).
    private static func captures(_ pattern: String, in line: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
              match.numberOfRanges > 1 else { return nil }
        return (1..<match.numberOfRanges).map { i in
            guard let range = Range(match.range(at: i), in: line) else { return "" }
            return String(line[range])
        }
    }
}
