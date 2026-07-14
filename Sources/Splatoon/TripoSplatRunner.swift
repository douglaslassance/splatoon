import Foundation

// MARK: - TripoSplat single-image generator (via the tripo-cli tool)
//
// Turns one photo into a complete 3D Gaussian object (not SHARP's 2.5D relief) by
// shelling out to `tripo-cli generate <image> <out.ply>` — a Python/PyTorch
// wrapper around TripoSplat that runs on Apple Silicon (MPS). It writes a standard
// 3DGS PLY, so the viewer and meshers consume it unchanged. Blocking; drive from a
// background task. Cancellable.

final class TripoSplatRunner {

    enum RunError: LocalizedError {
        case toolFailed(code: Int32, log: String)
        case noOutput
        case cancelled

        var errorDescription: String? {
            switch self {
            case .toolFailed(let code, let log):
                // tripo-cli prints a clear message when weights are missing.
                if log.contains("tripo-cli download") {
                    return "TripoSplat's model weights aren't downloaded. Run `tripo-cli download`, then try again."
                }
                return "TripoSplat failed (exit \(code)).\n\(log)"
            case .noOutput:  return "TripoSplat produced no splat."
            case .cancelled: return "Generation cancelled."
            }
        }
    }

    private let tool: URL
    /// Number of Gaussians to generate (passed to `--num-gaussians`).
    var gaussians = 131072

    private let lock = NSLock()
    private var currentProcess: Process?
    private var cancelled = false

    init?() {
        guard let tool = ToolLocator.resolvedURL(for: .tripoSplat) else { return nil }
        self.tool = tool
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

    /// Generate a splat PLY at `output` from the image at `imagePath`. `progress`
    /// reports a stage label + overall 0…1 fraction (dispatch to the UI is the
    /// caller's job). Blocking; throws on failure.
    func run(imagePath: URL, output: URL, workDir: URL,
             progress: @escaping (_ label: String, _ fraction: Double) -> Void) throws {
        let fm = FileManager.default
        try? fm.createDirectory(at: workDir, withIntermediateDirectories: true)
        try? fm.removeItem(at: output)

        progress("Loading model…", 0.05)
        // Diffusion sampling ("Sampling: N%") is the bulk of the time; map it to
        // most of the bar, leaving a head for load and a tail for decode/write.
        try runTool(args: [
            "generate", imagePath.path, output.path,
            "--device", "mps",
            "--num-gaussians", String(gaussians),
        ], workDir: workDir) { line in
            if let caps = Self.captures(#"Sampling:\s*(\d+)%"#, in: line), let pct = Double(caps[0]) {
                progress("Generating 3D object…", 0.10 + 0.80 * (pct / 100))
            } else if line.contains("Decoding") || line.contains("save") {
                progress("Finishing…", 0.95)
            }
        }

        guard fm.fileExists(atPath: output.path) else { throw RunError.noOutput }
        progress("Finishing…", 1.0)
    }

    // MARK: - Subprocess (mirrors the reconstruction pipeline's runTool)

    private func runTool(args: [String], workDir: URL, onLine: @escaping (String) -> Void) throws {
        if isCancelled { throw RunError.cancelled }

        let fm = FileManager.default
        try? fm.createDirectory(at: workDir, withIntermediateDirectories: true)
        let logURL = workDir.appendingPathComponent("triposplat.log")
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
            lineBuffer.append(data)
            // tqdm redraws progress with carriage returns, so split on \r and \n.
            while let idx = lineBuffer.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
                let lineData = lineBuffer[lineBuffer.startIndex..<idx]
                lineBuffer.removeSubrange(lineBuffer.startIndex...idx)
                if let line = String(data: lineData, encoding: .utf8) { onLine(line) }
            }
        }

        let process = Process()
        process.executableURL = tool
        process.arguments = args
        process.currentDirectoryURL = workDir
        process.standardOutput = pipe
        process.standardError = pipe

        lock.lock(); currentProcess = process; lock.unlock()

        do {
            try process.run()
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            throw RunError.toolFailed(code: -1, log: error.localizedDescription)
        }
        process.waitUntilExit()
        drained.wait()

        lock.lock(); currentProcess = nil; lock.unlock()

        if isCancelled { throw RunError.cancelled }
        if process.terminationStatus != 0 {
            let log = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
            throw RunError.toolFailed(code: process.terminationStatus, log: String(log.suffix(1500)))
        }
    }

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
