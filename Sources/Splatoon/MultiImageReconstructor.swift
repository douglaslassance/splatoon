import Foundation

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
        case noOutput
        case cancelled

        var errorDescription: String? {
            switch self {
            case .toolFailed(let tool, let stage, let code, let log):
                return "\(tool) failed during \(stage) (exit \(code)).\n\(log)"
            case .noSparseModel:
                return "COLMAP could not solve camera poses. The photos may lack enough overlap or texture."
            case .noOutput:
                return "The trainer produced no splat file."
            case .cancelled:
                return "Reconstruction cancelled."
            }
        }
    }

    private let colmap: URL
    private let trainer: URL

    /// Number of OpenSplat training iterations. Higher = sharper, slower.
    var trainingIterations = 2000

    private let lock = NSLock()
    private var currentProcess: Process?
    private var cancelled = false

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
    /// `workDir` is scratch space; the splat is written to `output`. `progress`
    /// reports human-readable stage messages (dispatch to the UI is the caller's job).
    func run(imagesDir: URL, workDir: URL, output: URL, progress: (String) -> Void) throws {
        let fm = FileManager.default
        let databaseURL = workDir.appendingPathComponent("database.db")
        let sparseDir = workDir.appendingPathComponent("sparse", isDirectory: true)
        try? fm.createDirectory(at: sparseDir, withIntermediateDirectories: true)

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
        progress("Detecting features…")
        try runTool(colmap, stage: "feature extraction", workDir: workDir, args: [
            "feature_extractor",
            "--database_path", databaseURL.path,
            "--image_path", imagesDir.path,
            "--ImageReader.single_camera", "1",
            extractGPU, "0",
        ])

        // 2. Match features across all image pairs.
        progress("Matching images…")
        try runTool(colmap, stage: "matching", workDir: workDir, args: [
            "exhaustive_matcher",
            "--database_path", databaseURL.path,
            matchGPU, "0",
        ])

        // 3. Sparse reconstruction (camera poses + sparse points).
        progress("Solving camera poses…")
        try runTool(colmap, stage: "mapping", workDir: workDir, args: [
            "mapper",
            "--database_path", databaseURL.path,
            "--image_path", imagesDir.path,
            "--output_path", sparseDir.path,
        ])

        // COLMAP writes one model per subfolder (0, 1, …); model 0 is the largest.
        let model0 = sparseDir.appendingPathComponent("0", isDirectory: true)
        guard fm.fileExists(atPath: model0.appendingPathComponent("cameras.bin").path)
                || fm.fileExists(atPath: model0.appendingPathComponent("cameras.txt").path) else {
            throw ReconstructionError.noSparseModel
        }

        // OpenSplat expects a COLMAP project laid out as <dir>/images and
        // <dir>/sparse/0. workDir already has sparse/0; symlink images in.
        let projectImages = workDir.appendingPathComponent("images", isDirectory: true)
        if !fm.fileExists(atPath: projectImages.path) {
            try? fm.createSymbolicLink(at: projectImages, withDestinationURL: imagesDir)
        }

        // 4. Train the Gaussian splat.
        progress("Training splat…")
        try runTool(trainer, stage: "training", workDir: workDir, args: [
            workDir.path,
            "-n", String(trainingIterations),
            "-o", output.path,
        ])

        guard fm.fileExists(atPath: output.path) else { throw ReconstructionError.noOutput }
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

    /// Run one tool to completion, streaming its output to `<stage>.log` (a file,
    /// so a full pipe buffer can never deadlock us). Throws on nonzero exit.
    private func runTool(_ executable: URL, stage: String, workDir: URL, args: [String]) throws {
        if isCancelled { throw ReconstructionError.cancelled }

        let fm = FileManager.default
        let logURL = workDir.appendingPathComponent("\(stage.replacingOccurrences(of: " ", with: "_")).log")
        fm.createFile(atPath: logURL.path, contents: nil)
        let logHandle = try FileHandle(forWritingTo: logURL)
        defer { try? logHandle.close() }

        let process = Process()
        process.executableURL = executable
        process.arguments = args
        process.currentDirectoryURL = workDir
        process.standardOutput = logHandle
        process.standardError = logHandle

        lock.lock(); currentProcess = process; lock.unlock()

        do {
            try process.run()
        } catch {
            throw ReconstructionError.toolFailed(tool: executable.lastPathComponent, stage: stage,
                                                 code: -1, log: error.localizedDescription)
        }
        process.waitUntilExit()

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
