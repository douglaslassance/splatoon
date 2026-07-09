import Foundation
import simd

/// A starting viewpoint for `SplatViewer`'s fly camera, in the same
/// post-calibration (OpenCV -> OpenGL flipped) world space it already renders
/// in — see `SplatViewer`'s `calibration` matrix.
struct ScenePose: Equatable {
    var eye: SIMD3<Float>
    var yaw: Float
    var pitch: Float
}

/// Reconstruction progress: a human stage label plus how far through the
/// COLMAP+training pipeline we are, 0...1. Weighted toward training, which
/// dominates wall time but is also the only stage with a precise counter
/// (OpenSplat prints `Step N: loss (X%)` for every step).
struct ReconstructionProgress: Equatable {
    var stageLabel: String
    var fraction: Double
}

/// Runs the multi-image reconstruction pipeline by shelling out to COLMAP (sparse
/// pose solve) then OpenSplat (Gaussian-splat training). Blocking; drive it from a
/// background task. Cancellable: `cancel()` terminates the running subprocess.
///
/// The app is not sandboxed, so spawning these subprocesses and using a temp
/// working directory is unrestricted.
final class MultiImageReconstructor {

    enum ReconstructionError: LocalizedError {
        case toolFailed(tool: String, stage: String, code: Int32, log: String)
        case noSparseModel
        case insufficientRegistration(registered: Int, total: Int)
        case noOutput
        case cancelled

        var errorDescription: String? {
            switch self {
            case .toolFailed(let tool, let stage, let code, let log):
                return "\(tool) failed during \(stage) (exit \(code)).\n\(log)"
            case .noSparseModel:
                return "COLMAP couldn't align the photos into a scene — they likely don't overlap enough, or "
                    + "lack texture. Capture a slow, continuous orbit where each photo shares most of its view "
                    + "with the next."
            case .insufficientRegistration(let registered, let total):
                return "Only \(registered) of \(total) photos could be aligned, too few for a good result. "
                    + "The photos likely don't overlap enough — capture a slow, continuous orbit where each "
                    + "photo shares most of its view with the next, rather than a few spread-out angles."
            case .noOutput:
                return "The trainer produced no splat file."
            case .cancelled:
                return "Reconstruction cancelled."
            }
        }
    }

    /// One COLMAP sparse model and how many images it registered.
    private struct SparseModel { let url: URL; let registered: Int }

    private let colmap: URL
    private let trainer: URL

    /// Number of OpenSplat training iterations. Higher = sharper, slower.
    var trainingIterations = 2000

    private let lock = NSLock()
    private var currentProcess: Process?
    private var cancelled = false
    private var pidLockURL: URL?

    init(colmap: URL, trainer: URL) {
        self.colmap = colmap
        self.trainer = trainer
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

    /// Run COLMAP + OpenSplat end to end. `imagesDir` holds the prepared photos;
    /// `workDir` is scratch space; the splat is written to `output`. `totalImages`
    /// is the pose-solving denominator for the mapping stage's live progress.
    /// `progress` reports a stage label and an overall 0...1 fraction across the
    /// whole pipeline (dispatch to the UI is the caller's job).
    func run(imagesDir: URL, workDir: URL, output: URL, totalImages: Int,
            progress: @escaping (ReconstructionProgress) -> Void) throws {
        // An orphaned subprocess from a previous run (app crashed, was
        // force-quit, or updated mid-reconstruction) doesn't die with its
        // parent on Unix — it keeps running and would race a fresh run over
        // this exact scene's output file. Clear it before starting.
        Self.killStaleProcess(for: output)
        let lockURL = Self.pidLockURL(for: output)
        self.pidLockURL = lockURL
        defer {
            self.pidLockURL = nil
            try? FileManager.default.removeItem(at: lockURL)
        }

        let fm = FileManager.default
        let databaseURL = workDir.appendingPathComponent("database.db")
        let sparseDir = workDir.appendingPathComponent("sparse", isDirectory: true)
        try? fm.createDirectory(at: sparseDir, withIntermediateDirectories: true)

        // COLMAP stages are quick (roughly a minute for a handful of photos);
        // training dominates wall time and gets the bulk of the bar, and is the
        // only stage OpenSplat reports a precise percentage for.
        let bands: [(stage: String, start: Double, end: Double)] = [
            ("Detecting features…", 0.00, 0.10),
            ("Matching images…", 0.10, 0.15),
            ("Solving camera poses…", 0.15, 0.25),
            ("Training splat…", 0.25, 1.00),
        ]
        func report(_ band: (stage: String, start: Double, end: Double), within: Double) {
            let f = band.start + (band.end - band.start) * max(0, min(1, within))
            progress(ReconstructionProgress(stageLabel: band.stage, fraction: f))
        }

        // COLMAP renamed the CPU/GPU toggles across versions, and its option
        // parser hard-errors on an unknown flag. Probe --help and pass whichever
        // this build accepts. Forcing CPU keeps it working headlessly (GPU SIFT
        // needs a GL context) and on Mac builds compiled without CUDA.
        let extractGPU = gpuFlag(subcommand: "feature_extractor",
                                 preferred: "--FeatureExtraction.use_gpu",
                                 legacy: "--SiftExtraction.use_gpu")
        let matchGPU = gpuFlag(subcommand: "exhaustive_matcher",
                               preferred: "--FeatureMatching.use_gpu",
                               legacy: "--SiftMatching.use_gpu")

        // 1. Detect image features.
        report(bands[0], within: 0)
        try runTool(colmap, stage: "feature extraction", workDir: workDir, args: [
            "feature_extractor",
            "--database_path", databaseURL.path,
            "--image_path", imagesDir.path,
            "--ImageReader.single_camera", "1",
            extractGPU, "0",
        ]) { line in
            guard let caps = Self.captures(#"Processed file \[(\d+)/(\d+)\]"#, in: line),
                  let n = Double(caps[0]), let total = Double(caps[1]), total > 0 else { return }
            report(bands[0], within: n / total)
        }

        // 2. Match features across all image pairs.
        report(bands[1], within: 0)
        try runTool(colmap, stage: "matching", workDir: workDir, args: [
            "exhaustive_matcher",
            "--database_path", databaseURL.path,
            matchGPU, "0",
        ]) { line in
            guard let caps = Self.captures(#"Processing block \[(\d+)/(\d+),"#, in: line),
                  let n = Double(caps[0]), let total = Double(caps[1]), total > 0 else { return }
            report(bands[1], within: n / total)
        }

        // 3. Sparse reconstruction (camera poses + sparse points). When the
        // photos can't be connected, the mapper produces no model and exits
        // nonzero — surface the friendly overlap guidance, not a raw COLMAP log.
        report(bands[2], within: 0)
        do {
            try runTool(colmap, stage: "mapping", workDir: workDir, args: [
                "mapper",
                "--database_path", databaseURL.path,
                "--image_path", imagesDir.path,
                "--output_path", sparseDir.path,
            ]) { line in
                guard totalImages > 0, let caps = Self.captures(#"num_reg_frames=(\d+)"#, in: line),
                      let n = Double(caps[0]) else { return }
                report(bands[2], within: n / Double(totalImages))
            }
        } catch ReconstructionError.toolFailed {
            throw ReconstructionError.noSparseModel   // (cancellation still propagates)
        }

        // When COLMAP can't connect all the photos into one model, it fragments
        // into several (sparse/0, sparse/1, …) — and `0` is NOT guaranteed to be
        // the largest. Pick the model with the most registered images so we train
        // on the best-connected subset, not a stray 2-image fragment.
        let models = Self.sparseModels(in: sparseDir)
        guard let best = models.max(by: { $0.registered < $1.registered }) else {
            throw ReconstructionError.noSparseModel
        }

        // Guard against the silent-garbage case: training on only a couple of
        // views produces floater soup after a long wait. Fail fast, before the
        // expensive training step, with capture guidance the user can act on.
        let needed = max(3, Int(ceil(Double(totalImages) * 0.5)))
        guard best.registered >= needed else {
            throw ReconstructionError.insufficientRegistration(registered: best.registered, total: totalImages)
        }

        // OpenSplat reads the COLMAP project's sparse/0, so make the best model
        // be sparse/0 (renaming the fragments out of the way if it isn't already).
        let zero = sparseDir.appendingPathComponent("0", isDirectory: true)
        if best.url.lastPathComponent != "0" {
            let displaced = sparseDir.appendingPathComponent("0_replaced", isDirectory: true)
            try? fm.removeItem(at: displaced)
            if fm.fileExists(atPath: zero.path) { try? fm.moveItem(at: zero, to: displaced) }
            try? fm.moveItem(at: best.url, to: zero)
        }

        // OpenSplat expects a COLMAP project laid out as <dir>/images and
        // <dir>/sparse/0. workDir already has sparse/0; symlink images in.
        let projectImages = workDir.appendingPathComponent("images", isDirectory: true)
        if !fm.fileExists(atPath: projectImages.path) {
            try? fm.createSymbolicLink(at: projectImages, withDestinationURL: imagesDir)
        }

        // 4. Train the Gaussian splat. OpenSplat prints its own percentage
        // (relative to -n), already exactly what we want for `within`.
        report(bands[3], within: 0)
        try runTool(trainer, stage: "training", workDir: workDir, args: [
            workDir.path,
            "-n", String(trainingIterations),
            "-o", output.path,
        ]) { line in
            guard let caps = Self.captures(#"Step \d+: [^(]*\((\d+)%\)"#, in: line),
                  let percent = Double(caps[0]) else { return }
            report(bands[3], within: percent / 100)
        }

        guard fm.fileExists(atPath: output.path) else { throw ReconstructionError.noOutput }

        // OpenSplat also writes cameras.json next to `output`, always under that
        // fixed name — move it to a name scoped to this scene before it collides
        // with the next reconstruction's cameras.json in the same cache directory.
        let writtenCameras = output.deletingLastPathComponent().appendingPathComponent("cameras.json")
        if fm.fileExists(atPath: writtenCameras.path) {
            try? fm.removeItem(at: Self.camerasURL(for: output))
            try? fm.moveItem(at: writtenCameras, to: Self.camerasURL(for: output))
        }
    }

    /// The registered camera poses OpenSplat trained against, scoped to `plyURL`.
    static func camerasURL(for plyURL: URL) -> URL {
        plyURL.deletingPathExtension().appendingPathExtension("cameras.json")
    }

    /// Every COLMAP sparse model under `sparseDir` (each numbered subfolder that
    /// holds a reconstruction), with its registered-image count.
    private static func sparseModels(in sparseDir: URL) -> [SparseModel] {
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(at: sparseDir, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
        return entries.compactMap { url in
            let hasModel = fm.fileExists(atPath: url.appendingPathComponent("images.bin").path)
                || fm.fileExists(atPath: url.appendingPathComponent("images.txt").path)
            guard hasModel else { return nil }
            return SparseModel(url: url, registered: registeredImageCount(in: url))
        }
    }

    /// How many images a COLMAP model registered. `images.bin` begins with a
    /// little-endian UInt64 image count; fall back to the `# Number of images`
    /// header if only the text form exists.
    private static func registeredImageCount(in modelDir: URL) -> Int {
        let binURL = modelDir.appendingPathComponent("images.bin")
        if let handle = try? FileHandle(forReadingFrom: binURL) {
            defer { try? handle.close() }
            if let data = try? handle.read(upToCount: 8), data.count == 8 {
                return Int(UInt64(littleEndian: data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }))
            }
        }
        if let text = try? String(contentsOf: modelDir.appendingPathComponent("images.txt"), encoding: .utf8),
           let caps = captures(#"Number of images: (\d+)"#, in: text), let n = Int(caps[0]) {
            return n
        }
        return 0
    }

    /// Where the currently-active subprocess's PID is recorded for this scene's
    /// output, scoped exactly like `camerasURL(for:)`.
    private static func pidLockURL(for output: URL) -> URL {
        output.deletingPathExtension().appendingPathExtension("pid")
    }

    /// If a lock file exists for `output` and still names a live process, kill
    /// it and remove the lock. Call before starting a fresh reconstruction for
    /// the same scene, so an orphan from a previous run can't race a new one
    /// over the same output file.
    static func killStaleProcess(for output: URL) {
        let lockURL = pidLockURL(for: output)
        let fm = FileManager.default
        defer { try? fm.removeItem(at: lockURL) }
        guard let text = try? String(contentsOf: lockURL, encoding: .utf8),
              let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)) else { return }
        // Verify it's actually one of our tools before killing — PIDs get
        // reused, so a stale number could coincidentally belong to something
        // unrelated by the time we check.
        guard isRunning(pid) else { return }
        let command = commandLine(of: pid)
        guard command.contains("opensplat") || command.contains("colmap") else { return }
        kill(pid, SIGKILL)
    }

    private static func isRunning(_ pid: pid_t) -> Bool {
        kill(pid, 0) == 0
    }

    private static func commandLine(of pid: pid_t) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-o", "command=", "-p", String(pid)]
        let pipe = Pipe()
        process.standardOutput = pipe
        guard (try? process.run()) != nil else { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(data: data, encoding: .utf8) ?? ""
    }

    /// Reads the scene's first registered camera and converts it into
    /// `SplatViewer`'s (eye, yaw, pitch) convention, so a scene splat opens
    /// looking at what was actually photographed instead of world origin (which
    /// is only meaningful for SHARP's single-image "eye = the photo" convention;
    /// COLMAP's world origin is an arbitrary artifact of its bootstrap pair).
    static func initialPose(for plyURL: URL) -> ScenePose? {
        struct Camera: Decodable {
            let position: [Float]
            let rotation: [[Float]]
        }
        guard let data = try? Data(contentsOf: camerasURL(for: plyURL)),
              let cameras = try? JSONDecoder().decode([Camera].self, from: data),
              let cam = cameras.first, cam.position.count == 3, cam.rotation.count == 3 else {
            return nil
        }

        // cameras.json stores camera-to-world in OpenCV convention (X right, Y
        // down, Z forward), row-major — the same convention raw PLY positions
        // use. Flip into the post-calibration space SplatViewer renders in.
        func flip(_ v: SIMD3<Float>) -> SIMD3<Float> { SIMD3(v.x, -v.y, -v.z) }

        let eye = flip(SIMD3(cam.position[0], cam.position[1], cam.position[2]))
        // Column 2 of the camera-to-world rotation = the camera's local +Z
        // (forward) axis expressed in world space.
        let forwardRaw = SIMD3<Float>(cam.rotation[0][2], cam.rotation[1][2], cam.rotation[2][2])
        let forward = flip(forwardRaw)

        // Invert SplatViewer's own forwardVector(yaw,pitch) formula.
        let pitch = asin(max(-1, min(1, forward.y)))
        let yaw = atan2(forward.x, -forward.z)
        return ScenePose(eye: eye, yaw: yaw, pitch: pitch)
    }

    /// The first match's capture groups as strings, or nil if `pattern` doesn't
    /// match `line`. `pattern` is a raw NSRegularExpression pattern (not a Swift
    /// regex literal — those parse ambiguously as division in some call-site
    /// positions in this file).
    private static func captures(_ pattern: String, in line: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
              match.numberOfRanges > 1 else { return nil }
        return (1..<match.numberOfRanges).map { i in
            guard let range = Range(match.range(at: i), in: line) else { return "" }
            return String(line[range])
        }
    }

    /// Pick the option name this COLMAP build understands, preferring the newer
    /// spelling and falling back to the legacy one.
    private func gpuFlag(subcommand: String, preferred: String, legacy: String) -> String {
        helpText(colmap, subcommand).contains(preferred) ? preferred : legacy
    }

    /// Capture a subcommand's `--help` text (small; safe to read fully).
    private func helpText(_ executable: URL, _ subcommand: String) -> String {
        let process = Process()
        process.executableURL = executable
        process.arguments = [subcommand, "--help"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        guard (try? process.run()) != nil else { return "" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(data: data, encoding: .utf8) ?? ""
    }

    // MARK: - Subprocess

    /// Run one tool to completion. Output streams through a `Pipe`, drained live
    /// by a `readabilityHandler` (so a full pipe buffer can never deadlock us,
    /// same guarantee the old file-redirect had) — every byte is both appended to
    /// `<stage>.log` for post-mortem debugging and, line by line, handed to
    /// `onLine` for live progress parsing. Throws on nonzero exit.
    private func runTool(_ executable: URL, stage: String, workDir: URL, args: [String],
                         onLine: ((String) -> Void)? = nil) throws {
        if isCancelled { throw ReconstructionError.cancelled }

        let fm = FileManager.default
        let logURL = workDir.appendingPathComponent("\(stage.replacingOccurrences(of: " ", with: "_")).log")
        fm.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try FileHandle(forWritingTo: logURL)
        defer { try? logHandle.close() }

        let pipe = Pipe()
        var lineBuffer = Data()
        // `availableData` returns empty Data exactly at EOF (once the child exits
        // and its copy of the write end closes) — a reliable "fully drained"
        // signal, unlike racing this against `waitUntilExit()`'s return time.
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
            while let newline = lineBuffer.firstIndex(of: 0x0A) {
                let lineData = lineBuffer[lineBuffer.startIndex..<newline]
                lineBuffer.removeSubrange(lineBuffer.startIndex...newline)
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
            throw ReconstructionError.toolFailed(tool: executable.lastPathComponent, stage: stage,
                                                 code: -1, log: error.localizedDescription)
        }
        // Record the now-running subprocess's PID so a future run can detect
        // and kill it if this one dies without cleaning up after itself.
        if let pidLockURL {
            try? String(process.processIdentifier).write(to: pidLockURL, atomically: true, encoding: .utf8)
        }
        process.waitUntilExit()
        drained.wait()   // let the handler finish draining/logging trailing output

        lock.lock(); currentProcess = nil; lock.unlock()

        if isCancelled { throw ReconstructionError.cancelled }
        if process.terminationStatus != 0 {
            let log = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
            throw ReconstructionError.toolFailed(tool: executable.lastPathComponent, stage: stage,
                                                 code: process.terminationStatus,
                                                 log: String(log.suffix(1500)))
        }
    }
}
