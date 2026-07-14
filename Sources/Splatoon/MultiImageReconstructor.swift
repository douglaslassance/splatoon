import Foundation
import simd

/// A starting viewpoint for `SplatViewer`'s fly camera, in the same
/// post-calibration (OpenCV -> OpenGL flipped) world space it already renders
/// in — see `SplatViewer`'s `calibration` matrix.
struct ScenePose: Equatable {
    var eye: SIMD3<Float>
    var yaw: Float
    var pitch: Float
    /// Vertical field of view of the registered camera, so the scene opens at the
    /// same framing the photo was taken with.
    var fovyDegrees: Float
}

/// SHARP's capture FOV: focal 1536 px on a 1536-px-tall square input →
/// 2·atan(height / 2·focal) = 2·atan(0.5) ≈ 53.13°. Used as the single-image
/// viewer's field of view so a splat opens framed like its source photo.
let sharpFOVyDegrees = Float(2 * atan(0.5) * 180 / Double.pi)

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

    /// Spherical-harmonics degree OpenSplat trains (1…3). Lower shrinks the splat
    /// (fewer colour coefficients per gaussian) and speeds rendering.
    var shDegree = 1

    /// Solve poses with COLMAP's global SfM (`global_mapper`, the upstreamed
    /// successor to the now-deprecated GLOMAP) instead of the incremental
    /// `mapper`. More robust on sparse, weakly-overlapping captures. Falls back to
    /// `mapper` if this COLMAP build predates `global_mapper`.
    var useGlobalSolver = false

    /// Which trainer runs step 4. `.openSplat` uses the `trainer` (opensplat)
    /// binary; `.brush` uses `brushBinary`. When `.brush` but `brushBinary` is
    /// nil, training falls back to OpenSplat.
    var trainerKind: SplatTrainer = .openSplat
    /// The `brush-cli` binary, resolved only when the Brush trainer is selected.
    var brushBinary: URL?

    private let lock = NSLock()
    private var currentProcess: Process?
    private var cancelled = false
    private var pidLockURL: URL?
    /// The active run's progress callback, stashed so the per-trainer helpers can
    /// report without threading the closure through every call. Set in `run`.
    private var progressSink: ((ReconstructionProgress) -> Void)?

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
    /// `sharedCamera` assumes one shared intrinsic across all frames (right for a
    /// single video or a one-phone photo group); pass false for mixed or
    /// multi-clip scenes so COLMAP solves intrinsics per image. `progress` reports
    /// a stage label and an overall 0...1 fraction across the whole pipeline
    /// (dispatch to the UI is the caller's job).
    func run(imagesDir: URL, workDir: URL, output: URL, totalImages: Int, sharedCamera: Bool = true,
            sequentialMatching: Bool = false,
            progress: @escaping (ReconstructionProgress) -> Void) throws {
        // An orphaned subprocess from a previous run (app crashed, was
        // force-quit, or updated mid-reconstruction) doesn't die with its
        // parent on Unix — it keeps running and would race a fresh run over
        // this exact scene's output file. Clear it before starting.
        Self.killStaleProcess(for: output)
        let lockURL = Self.pidLockURL(for: output)
        self.pidLockURL = lockURL
        self.progressSink = progress
        defer {
            self.pidLockURL = nil
            self.progressSink = nil
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
        // Sequential matching (temporal neighbours, O(n)) suits ordered video
        // frames; exhaustive (all pairs, O(n²)) is needed for unordered photos.
        let matcher = sequentialMatching ? "sequential_matcher" : "exhaustive_matcher"
        let matchGPU = gpuFlag(subcommand: matcher,
                               preferred: "--FeatureMatching.use_gpu",
                               legacy: "--SiftMatching.use_gpu")

        // 1. Detect image features.
        report(bands[0], within: 0)
        try runTool(colmap, stage: "feature extraction", workDir: workDir, args: [
            "feature_extractor",
            "--database_path", databaseURL.path,
            "--image_path", imagesDir.path,
            "--ImageReader.single_camera", sharedCamera ? "1" : "0",
            extractGPU, "0",
        ]) { line in
            guard let caps = Self.captures(#"Processed file \[(\d+)/(\d+)\]"#, in: line),
                  let n = Double(caps[0]), let total = Double(caps[1]), total > 0 else { return }
            report(bands[0], within: n / total)
        }

        // 2. Match features. Exhaustive reports "Processing block [n/total]";
        // sequential reports "Matching image [n/total]" — parse either for progress.
        report(bands[1], within: 0)
        try runTool(colmap, stage: "matching", workDir: workDir, args: [
            matcher,
            "--database_path", databaseURL.path,
            matchGPU, "0",
        ]) { line in
            guard let caps = Self.captures(#"(?:Processing block|Matching image) \[(\d+)/(\d+)"#, in: line),
                  let n = Double(caps[0]), let total = Double(caps[1]), total > 0 else { return }
            report(bands[1], within: n / total)
        }

        // 3. Sparse reconstruction (camera poses + sparse points). When the
        // photos can't be connected, the mapper produces no model and exits
        // nonzero — surface the friendly overlap guidance, not a raw COLMAP log.
        //
        // COLMAP's global SfM (`global_mapper`) is far more robust than the
        // incremental `mapper` on sparse, weakly-overlapping captures, where the
        // incremental solver often registers only a fraction of the views. It's
        // the same binary and takes the same args, writing the same sparse/<n>
        // model layout — so it's a drop-in for this stage. Used when the user opts
        // in, guarded by a support probe so an older COLMAP falls back gracefully.
        let subcommand = useGlobalSolver && colmapSupports("global_mapper") ? "global_mapper" : "mapper"
        report(bands[2], within: 0)
        do {
            try runTool(colmap, stage: "mapping", workDir: workDir, args: [
                subcommand,
                "--database_path", databaseURL.path,
                "--image_path", imagesDir.path,
                "--output_path", sparseDir.path,
            ]) { line in
                // Incremental mapper prints per-image registration progress; the
                // global one doesn't, so its band holds at the start until training.
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

        // Guard against the silent-garbage case: a marginal solve that registers
        // only part of the capture produces floater soup after a long wait. Fail
        // fast, before the expensive training step, with capture guidance the user
        // can act on. 60% is deliberately stricter than a bare majority — a
        // half-registered scene looks bad enough that guidance beats shipping it.
        let needed = max(3, Int(ceil(Double(totalImages) * 0.6)))
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

        // 4. Train the Gaussian splat. Both trainers read this COLMAP project and
        // write a standard 3DGS PLY to `output`; each also leaves a scene-scoped
        // cameras.json next to it (OpenSplat writes one; for Brush we synthesize
        // it from the COLMAP model) for pose recovery + gravity alignment.
        let trainBand = bands[3]
        report(trainBand, within: 0)
        let zeroModel = sparseDir.appendingPathComponent("0", isDirectory: true)
        if trainerKind == .brush, let brushBinary {
            try trainWithBrush(brushBinary, workDir: workDir, sparseZero: zeroModel,
                               output: output, band: trainBand)
        } else {
            try trainWithOpenSplat(workDir: workDir, output: output, band: trainBand)
        }

        guard fm.fileExists(atPath: output.path) else { throw ReconstructionError.noOutput }

        // The viewer's PLY reader only accepts a full (degree-3) or empty SH set;
        // a partial one (degree 1/2) fails to load and the viewport stays blank.
        // Strip partial sets to degree 0 so every splat we produce is loadable.
        Self.normalizeSHForViewer(output)

        // COLMAP's world frame is arbitrary (built off its bootstrap image pair),
        // so scenes come out tilted — the horizon isn't level and the fly camera
        // feels off. Rotate the splat + camera poses so the estimated gravity-up
        // becomes world-up, matching single-image splats' upright convention.
        Self.gravityAlign(plyURL: output, camerasURL: Self.camerasURL(for: output))
    }

    // MARK: - Trainers

    /// Train with OpenSplat (libtorch/MPS). Prints its own per-step percentage,
    /// exactly what `within` wants. Its Metal backend hard-crashes (SIGSEGV)
    /// during image loading on some builds — reliably so with the 1.1.5 Homebrew
    /// binary. Its CPU backend is reliable but much slower, so try GPU first and,
    /// if the trainer dies, retry once on CPU rather than failing after COLMAP.
    /// Moves OpenSplat's fixed-name cameras.json to this scene's scoped name.
    private func trainWithOpenSplat(workDir: URL, output: URL,
                                    band: (stage: String, start: Double, end: Double)) throws {
        func trainArgs(cpu: Bool) -> [String] {
            // --sh-degree must be >= 1 (OpenSplat rejects 0); clamp defensively.
            var args = [workDir.path, "-n", String(trainingIterations),
                        "--sh-degree", String(min(3, max(1, shDegree))), "-o", output.path]
            if cpu { args.append("--cpu") }
            return args
        }
        let cpuBand = (stage: "Training splat (CPU, slower)…", start: band.start, end: band.end)
        func report(_ b: (stage: String, start: Double, end: Double), within: Double) {
            let f = b.start + (b.end - b.start) * max(0, min(1, within))
            progressSink?(ReconstructionProgress(stageLabel: b.stage, fraction: f))
        }
        func onTrainLine(_ b: (stage: String, start: Double, end: Double)) -> (String) -> Void {
            { line in
                guard let caps = Self.captures(#"Step \d+: [^(]*\((\d+)%\)"#, in: line),
                      let percent = Double(caps[0]) else { return }
                report(b, within: percent / 100)
            }
        }

        do {
            try runTool(trainer, stage: "training", workDir: workDir,
                        args: trainArgs(cpu: false), onLine: onTrainLine(band))
        } catch ReconstructionError.toolFailed {
            // GPU trainer crashed. Don't retry if the user cancelled meanwhile.
            if isCancelled { throw ReconstructionError.cancelled }
            report(cpuBand, within: 0)
            try runTool(trainer, stage: "training (CPU)", workDir: workDir,
                        args: trainArgs(cpu: true), onLine: onTrainLine(cpuBand))
        }

        // OpenSplat writes cameras.json next to `output` under that fixed name —
        // move it to a scene-scoped name before the next reconstruction's collides.
        let fm = FileManager.default
        let writtenCameras = output.deletingLastPathComponent().appendingPathComponent("cameras.json")
        if fm.fileExists(atPath: writtenCameras.path) {
            try? fm.removeItem(at: Self.camerasURL(for: output))
            try? fm.moveItem(at: writtenCameras, to: Self.camerasURL(for: output))
        }
    }

    /// Train with Brush (native wgpu/Metal). Brush reads the COLMAP project
    /// directly and exports only a PLY, so we point its export at a scratch dir
    /// with a fixed name, move the result to `output`, then synthesize the
    /// cameras.json Brush doesn't write from the COLMAP model.
    private func trainWithBrush(_ brush: URL, workDir: URL, sparseZero: URL, output: URL,
                                band: (stage: String, start: Double, end: Double)) throws {
        let fm = FileManager.default
        let exportDir = workDir.appendingPathComponent("brush_export", isDirectory: true)
        try? fm.removeItem(at: exportDir)
        try? fm.createDirectory(at: exportDir, withIntermediateDirectories: true)

        // Brush loads the COLMAP project at `workDir` (viewer defaults off when a
        // source is given). sh-degree is 0…4 (unlike OpenSplat it allows 0). Brush
        // always exports on the final step, so a large --export-every just avoids
        // redundant intermediate writes; we take the newest PLY afterward.
        //
        // Stop densification at the halfway point so the second half of training
        // is a refinement/consolidation phase. Brush's default growth-stop (15000)
        // equals our default iteration count, which would run densification for
        // the entire training and skip refinement entirely — leaving a bloated,
        // over-bright splat cloud that washes the viewer white on harder scenes.
        let growthStop = max(1, trainingIterations / 2)
        let args = [
            workDir.path,
            "--total-train-iters", String(trainingIterations),
            "--growth-stop-iter", String(growthStop),
            "--sh-degree", String(min(4, max(0, shDegree))),
            "--max-resolution", "1920",
            "--export-path", exportDir.path,
            "--export-name", "splat.ply",
            "--export-every", String(trainingIterations),
        ]
        // RUST_LOG=info surfaces Brush's "Refine iter N" progress lines (its
        // logger is otherwise silent at the default error level).
        try runTool(brush, stage: "training", workDir: workDir, args: args,
                    env: ["RUST_LOG": "info"]) { [weak self] line in
            // Brush logs "Refine iter N, C splats." every refine step (~200 steps).
            guard let self, self.trainingIterations > 0,
                  let caps = Self.captures(#"Refine iter (\d+)"#, in: line),
                  let n = Double(caps[0]) else { return }
            let within = max(0, min(1, n / Double(self.trainingIterations)))
            self.progressSink?(ReconstructionProgress(stageLabel: band.stage,
                                                      fraction: band.start + (band.end - band.start) * within))
        }

        // Locate Brush's exported PLY (newest, in case it exported more than once)
        // and move it to `output`.
        let produced = ((try? fm.contentsOfDirectory(at: exportDir,
                                                     includingPropertiesForKeys: [.contentModificationDateKey])) ?? [])
            .filter { $0.pathExtension.lowercased() == "ply" }
            .max { (Self.modDate($0) ?? .distantPast) < (Self.modDate($1) ?? .distantPast) }
        guard let produced else { throw ReconstructionError.noOutput }
        try? fm.removeItem(at: output)
        try fm.moveItem(at: produced, to: output)

        // Brush leaves a few giant "background" splats (tens of world units) that
        // engulf the camera and wash the view flat; OpenSplat culls these during
        // training, Brush doesn't. Drop them before display.
        Self.cullOversizedSplats(output)

        // Brush writes no camera metadata; synthesize the cameras.json our pose
        // recovery + gravity alignment read, from the COLMAP model.
        writeCamerasJSON(fromColmapSparse: sparseZero, workDir: workDir,
                         to: Self.camerasURL(for: output))
    }

    /// MetalSplatter's PLY reader (the viewer's loader) hardcodes 45
    /// spherical-harmonics rest coefficients: it accepts either none (degree 0)
    /// or all 45 (degree 3), and throws on any partial set — so a degree-1 splat
    /// (9 coefficients) or degree-2 (24) fails to load and the viewport stays
    /// blank. Strip a partial set down to degree 0 (drop every `f_rest_*`
    /// property) so our output is always loadable. Degree-0 and degree-3 PLYs are
    /// left untouched. Best-effort: leaves the file alone if it can't be parsed.
    private static func normalizeSHForViewer(_ url: URL) {
        guard let data = try? Data(contentsOf: url),
              let headerRange = data.range(of: Data("end_header\n".utf8)),
              let header = String(data: data.subdata(in: 0..<headerRange.upperBound), encoding: .ascii) else {
            return
        }
        struct Prop { let name: String; let size: Int; let offset: Int }
        var props: [Prop] = []
        var format = "", count = 0, inVertex = false, offset = 0
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ").map(String.init)
            guard let keyword = parts.first else { continue }
            switch keyword {
            case "format" where parts.count >= 2:
                format = parts[1]
            case "element" where parts.count >= 3:
                inVertex = (parts[1] == "vertex")
                if inVertex { count = Int(parts[2]) ?? 0 }
            case "property" where inVertex && parts.count >= 3 && parts[1] != "list":
                let size = typeSize(parts[1])
                props.append(Prop(name: parts[2], size: size, offset: offset))
                offset += size
            default:
                break
            }
        }
        let stride = offset
        let restCount = props.filter { $0.name.hasPrefix("f_rest_") }.count
        // Only touch partial sets — degree 0 (none) and degree 3 (45) already load.
        guard format == "binary_little_endian", count > 0, stride > 0,
              restCount != 0, restCount != 45 else { return }
        let bodyStart = headerRange.upperBound
        guard bodyStart + count * stride <= data.count else { return }

        let keep = props.filter { !$0.name.hasPrefix("f_rest_") }
        let newStride = keep.reduce(0) { $0 + $1.size }
        var out = [UInt8](repeating: 0, count: count * newStride)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            out.withUnsafeMutableBytes { dst in
                for i in 0..<count {
                    var w = i * newStride
                    let base = bodyStart + i * stride
                    for p in keep {
                        memcpy(dst.baseAddress!.advanced(by: w),
                               raw.baseAddress!.advanced(by: base + p.offset), p.size)
                        w += p.size
                    }
                }
            }
        }
        // Rebuild the header without the f_rest_* property lines (preserving the
        // trailing newline after end_header via the empty final split element).
        let newHeader = header.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !($0.hasPrefix("property") && $0.contains(" f_rest_")) }
            .joined(separator: "\n")
        var result = Data(newHeader.utf8)
        result.append(contentsOf: out)
        try? result.write(to: url)
    }

    /// Drop pathologically large splats from a 3DGS PLY in place: any whose
    /// largest axis exceeds 50× the median splat's largest axis. That factor is
    /// scene-scale-invariant (it rides the splat-size distribution, not world
    /// units or the floater-inflated bounding box), so it removes only the handful
    /// of camera-engulfing outliers Brush leaves behind and never touches
    /// legitimate detail. Best-effort: leaves the file untouched if it can't parse
    /// the PLY, finds nothing to cull, or would cull everything.
    private static func cullOversizedSplats(_ url: URL) {
        guard let data = try? Data(contentsOf: url),
              let headerRange = data.range(of: Data("end_header\n".utf8)),
              let header = String(data: data.subdata(in: 0..<headerRange.upperBound), encoding: .ascii) else {
            return
        }
        var format = "", count = 0, inVertex = false, offset = 0
        var offsetOf: [String: Int] = [:]
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ").map(String.init)
            guard let keyword = parts.first else { continue }
            switch keyword {
            case "format" where parts.count >= 2:
                format = parts[1]
            case "element" where parts.count >= 3:
                inVertex = (parts[1] == "vertex")
                if inVertex { count = Int(parts[2]) ?? 0 }
            case "property" where inVertex && parts.count >= 3 && parts[1] != "list":
                offsetOf[parts[2]] = offset
                offset += typeSize(parts[1])
            default:
                break
            }
        }
        let stride = offset
        let bodyStart = headerRange.upperBound
        guard format == "binary_little_endian", count > 0, stride > 0,
              bodyStart + count * stride <= data.count,
              let s0 = offsetOf["scale_0"], let s1 = offsetOf["scale_1"], let s2 = offsetOf["scale_2"] else {
            return
        }

        var survivors: [Int] = []
        survivors.reserveCapacity(count)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            // Largest world-space axis of splat i (scales are stored logarithmic).
            func maxAxis(_ i: Int) -> Double {
                let base = bodyStart + i * stride
                let a = raw.loadUnaligned(fromByteOffset: base + s0, as: Float.self)
                let b = raw.loadUnaligned(fromByteOffset: base + s1, as: Float.self)
                let c = raw.loadUnaligned(fromByteOffset: base + s2, as: Float.self)
                return exp(Double(max(a, max(b, c))))
            }
            // Median over a bounded sample (exact median needs no more precision).
            var sample: [Double] = []
            let step = max(1, count / 200_000)
            var s = 0
            while s < count { sample.append(maxAxis(s)); s += step }
            sample.sort()
            let median = sample[sample.count / 2]
            let threshold = median * 50
            guard threshold.isFinite, threshold > 0 else { return }
            for i in 0..<count where maxAxis(i) <= threshold { survivors.append(i) }
        }
        // Nothing oversized, or (defensively) everything culled — leave as-is.
        guard survivors.count < count, !survivors.isEmpty else { return }

        var body = [UInt8](repeating: 0, count: survivors.count * stride)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            body.withUnsafeMutableBytes { dst in
                for (j, idx) in survivors.enumerated() {
                    memcpy(dst.baseAddress!.advanced(by: j * stride),
                           raw.baseAddress!.advanced(by: bodyStart + idx * stride), stride)
                }
            }
        }
        let newHeader = header.replacingOccurrences(of: "element vertex \(count)",
                                                    with: "element vertex \(survivors.count)")
        var result = Data(newHeader.utf8)
        result.append(contentsOf: body)
        try? result.write(to: url)
    }

    private static func modDate(_ url: URL) -> Date? {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
    }

    /// Build the OpenSplat-schema cameras.json our pose recovery + gravity
    /// alignment consume, straight from the COLMAP sparse model — used for the
    /// Brush path, which exports no camera metadata. Converts the binary model to
    /// text with `model_converter`, then maps each registered image's
    /// world-to-camera pose to the camera-to-world (position, rotation) form.
    /// Silent no-op on failure: cameras.json stays absent and `initialPose`
    /// falls back to opening at the origin.
    private func writeCamerasJSON(fromColmapSparse sparseZero: URL, workDir: URL, to camerasURL: URL) {
        let fm = FileManager.default
        let txtDir = workDir.appendingPathComponent("sparse_txt", isDirectory: true)
        try? fm.removeItem(at: txtDir)
        try? fm.createDirectory(at: txtDir, withIntermediateDirectories: true)
        do {
            try runTool(colmap, stage: "camera export", workDir: workDir, args: [
                "model_converter",
                "--input_path", sparseZero.path,
                "--output_path", txtDir.path,
                "--output_type", "TXT",
            ])
        } catch { return }
        guard let cameras = Self.parseColmapModel(txtDir: txtDir), !cameras.isEmpty,
              let data = try? JSONEncoder().encode(cameras) else { return }
        try? data.write(to: camerasURL)
    }

    /// Parse COLMAP's text `cameras.txt` + `images.txt` into cameras.json rows,
    /// ordered by image name (so the first row is the first captured frame, the
    /// pose `initialPose` opens at). Returns nil if the files can't be read.
    private static func parseColmapModel(txtDir: URL) -> [SceneCamera]? {
        guard let camerasText = try? String(contentsOf: txtDir.appendingPathComponent("cameras.txt"), encoding: .utf8),
              let imagesText = try? String(contentsOf: txtDir.appendingPathComponent("images.txt"), encoding: .utf8)
        else { return nil }

        // cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS…
        // Single-focal models put one focal in PARAMS[0]; the rest use fx,fy.
        let singleFocal: Set<String> = ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                                        "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV", "THIN_PRISM_FISHEYE"]
        struct Intrinsics { let width: Int; let height: Int; let fx: Float; let fy: Float }
        var intrinsics: [Int: Intrinsics] = [:]
        for line in camerasText.split(separator: "\n") where !line.hasPrefix("#") {
            let f = line.split(separator: " ").map(String.init)
            guard f.count >= 5, let id = Int(f[0]), let w = Int(f[2]), let h = Int(f[3]) else { continue }
            let params = f[4...].compactMap { Float($0) }
            guard let p0 = params.first else { continue }
            let fx = p0
            let fy = singleFocal.contains(f[1]) ? p0 : (params.count > 1 ? params[1] : p0)
            intrinsics[id] = Intrinsics(width: w, height: h, fx: fx, fy: fy)
        }

        // images.txt: two lines per image; the first is
        // IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME. Skip the second (points).
        var cameras: [SceneCamera] = []
        let imageLines = imagesText.split(separator: "\n", omittingEmptySubsequences: true)
        var i = 0
        while i < imageLines.count {
            let line = imageLines[i]
            if line.hasPrefix("#") { i += 1; continue }
            let f = line.split(separator: " ").map(String.init)
            // A pose line has ≥10 fields; its following points line is skipped.
            guard f.count >= 10,
                  let id = Int(f[0]),
                  let qw = Double(f[1]), let qx = Double(f[2]), let qy = Double(f[3]), let qz = Double(f[4]),
                  let tx = Double(f[5]), let ty = Double(f[6]), let tz = Double(f[7]),
                  let camID = Int(f[8]) else { i += 1; continue }
            let name = f[9...].joined(separator: " ")
            i += 2   // consume the pose line and its points line

            // COLMAP stores world-to-camera R (from quaternion) and t. cameras.json
            // wants camera-to-world: rotation = Rᵀ, position (camera centre) = −Rᵀt.
            let n = (qw*qw + qx*qx + qy*qy + qz*qz).squareRoot()
            guard n > 1e-9 else { continue }
            let w = qw/n, x = qx/n, y = qy/n, z = qz/n
            // R rows (world→camera).
            let r00 = 1 - 2*(y*y + z*z), r01 = 2*(x*y - z*w),     r02 = 2*(x*z + y*w)
            let r10 = 2*(x*y + z*w),     r11 = 1 - 2*(x*x + z*z), r12 = 2*(y*z - x*w)
            let r20 = 2*(x*z - y*w),     r21 = 2*(y*z + x*w),     r22 = 1 - 2*(x*x + y*y)
            // Rᵀ (camera→world), row-major.
            let rt: [[Float]] = [[Float(r00), Float(r10), Float(r20)],
                                 [Float(r01), Float(r11), Float(r21)],
                                 [Float(r02), Float(r12), Float(r22)]]
            // −Rᵀt.
            let cx = -(r00*tx + r10*ty + r20*tz)
            let cy = -(r01*tx + r11*ty + r21*tz)
            let cz = -(r02*tx + r12*ty + r22*tz)
            let intr = intrinsics[camID]
            cameras.append(SceneCamera(id: id, img_name: name, width: intr?.width, height: intr?.height,
                                       fx: intr?.fx, fy: intr?.fy,
                                       position: [Float(cx), Float(cy), Float(cz)], rotation: rt))
        }
        return cameras.sorted { ($0.img_name ?? "") < ($1.img_name ?? "") }
    }

    /// The registered camera poses OpenSplat trained against, scoped to `plyURL`.
    static func camerasURL(for plyURL: URL) -> URL {
        plyURL.deletingPathExtension().appendingPathExtension("cameras.json")
    }

    // MARK: - Gravity alignment

    /// One camera as stored in OpenSplat's cameras.json (all fields preserved so
    /// re-encoding after rotation doesn't drop anything downstream reads).
    private struct SceneCamera: Codable {
        var id: Int?
        var img_name: String?
        var width: Int?
        var height: Int?
        var fx: Float?
        var fy: Float?
        var position: [Float]
        var rotation: [[Float]]
    }

    /// Rotate the scene so its estimated gravity-up aligns with the app's upright
    /// convention. Estimates down as the average camera local-Y (OpenCV cameras
    /// look with +Y down); handheld captures are roughly upright, so this
    /// approximates gravity. Bakes the rotation into both `plyURL` and
    /// `camerasURL` so everything downstream stays consistent.
    static func gravityAlign(plyURL: URL, camerasURL: URL) {
        guard let data = try? Data(contentsOf: camerasURL),
              var cameras = try? JSONDecoder().decode([SceneCamera].self, from: data),
              !cameras.isEmpty else { return }

        var down = SIMD3<Float>(0, 0, 0)
        for cam in cameras where cam.rotation.count == 3 && cam.rotation.allSatisfy({ $0.count == 3 }) {
            // Camera-to-world column 1 = the camera's local +Y (down) in world.
            down += SIMD3(cam.rotation[0][1], cam.rotation[1][1], cam.rotation[2][1])
        }
        let length = simd_length(down)
        guard length > 1e-3 else { return }
        down /= length

        // Map estimated down -> (0,1,0). After the downstream OpenCV->OpenGL flip
        // (π about X, shared by SplatViewer's calibration and MeshExport), up then
        // lands on screen-up. Skip if the scene is already close to upright.
        let rotation = simd_quatf(from: down, to: SIMD3<Float>(0, 1, 0))
        guard rotation.angle.isFinite, abs(rotation.angle) > 0.02 else { return }

        guard rotatePLY(plyURL, by: rotation) else { return }

        for i in cameras.indices {
            if cameras[i].position.count == 3 {
                let p = simd_act(rotation, SIMD3(cameras[i].position[0], cameras[i].position[1], cameras[i].position[2]))
                cameras[i].position = [p.x, p.y, p.z]
            }
            if cameras[i].rotation.count == 3 {
                var r = cameras[i].rotation
                for col in 0..<3 {
                    let axis = simd_act(rotation, SIMD3(r[0][col], r[1][col], r[2][col]))
                    r[0][col] = axis.x; r[1][col] = axis.y; r[2][col] = axis.z
                }
                cameras[i].rotation = r
            }
        }
        if let encoded = try? JSONEncoder().encode(cameras) { try? encoded.write(to: camerasURL) }
    }

    /// Rotate a binary-little-endian 3DGS PLY in place: positions and normals as
    /// vectors, the rotation quaternion (stored w,x,y,z) composed with `rotation`.
    /// All other properties (colour, opacity, scale, SH bands) are untouched.
    /// Returns false if the PLY couldn't be parsed/rotated.
    private static func rotatePLY(_ url: URL, by rotation: simd_quatf) -> Bool {
        guard var data = try? Data(contentsOf: url),
              let headerRange = data.range(of: Data("end_header\n".utf8)),
              let header = String(data: data.subdata(in: 0..<headerRange.upperBound), encoding: .ascii) else {
            return false
        }
        var format = ""
        var count = 0
        var inVertex = false
        var offset = 0
        var offsetOf: [String: Int] = [:]
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ").map(String.init)
            guard let keyword = parts.first else { continue }
            switch keyword {
            case "format" where parts.count >= 2:
                format = parts[1]
            case "element" where parts.count >= 3:
                inVertex = (parts[1] == "vertex")
                if inVertex { count = Int(parts[2]) ?? 0 }
            case "property" where inVertex && parts.count >= 3 && parts[1] != "list":
                offsetOf[parts[2]] = offset
                offset += typeSize(parts[1])
            default:
                break
            }
        }
        let stride = offset
        let bodyStart = headerRange.upperBound
        guard format == "binary_little_endian", count > 0, stride > 0,
              bodyStart + count * stride <= data.count,
              let ox = offsetOf["x"], let oy = offsetOf["y"], let oz = offsetOf["z"] else {
            return false
        }
        let normal = (offsetOf["nx"], offsetOf["ny"], offsetOf["nz"])
        let quat = (offsetOf["rot_0"], offsetOf["rot_1"], offsetOf["rot_2"], offsetOf["rot_3"])

        data.withUnsafeMutableBytes { rawMut in
            let raw = UnsafeRawBufferPointer(rawMut)
            func load(_ off: Int) -> Float { raw.loadUnaligned(fromByteOffset: off, as: Float.self) }
            func store(_ value: Float, _ off: Int) {
                var v = value
                withUnsafeBytes(of: &v) { for k in 0..<4 { rawMut[off + k] = $0[k] } }
            }
            for i in 0..<count {
                let base = bodyStart + i * stride
                let p = simd_act(rotation, SIMD3(load(base + ox), load(base + oy), load(base + oz)))
                store(p.x, base + ox); store(p.y, base + oy); store(p.z, base + oz)

                if let nx = normal.0, let ny = normal.1, let nz = normal.2 {
                    let n = simd_act(rotation, SIMD3(load(base + nx), load(base + ny), load(base + nz)))
                    store(n.x, base + nx); store(n.y, base + ny); store(n.z, base + nz)
                }
                if let r0 = quat.0, let r1 = quat.1, let r2 = quat.2, let r3 = quat.3 {
                    // Stored (w, x, y, z); new orientation = rotation * old.
                    let old = simd_quatf(ix: load(base + r1), iy: load(base + r2), iz: load(base + r3), r: load(base + r0))
                    let composed = rotation * old
                    store(composed.real, base + r0)
                    store(composed.imag.x, base + r1)
                    store(composed.imag.y, base + r2)
                    store(composed.imag.z, base + r3)
                }
            }
        }
        do { try data.write(to: url); return true } catch { return false }
    }

    private static func typeSize(_ type: String) -> Int {
        switch type {
        case "char", "uchar", "int8", "uint8": return 1
        case "short", "ushort", "int16", "uint16": return 2
        case "double", "float64", "int64", "uint64": return 8
        default: return 4   // float/float32, int/uint/int32/uint32
        }
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
            let fy: Float?
            let height: Int?
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

        // Vertical FOV from the camera's calibrated focal length + image height,
        // so the scene opens framed like the photo. Fall back to SHARP's default.
        let fovy: Float
        if let fy = cam.fy, let h = cam.height, fy > 0 {
            fovy = Float(2 * atan(Double(h) / (2 * Double(fy))) * 180 / Double.pi)
        } else {
            fovy = sharpFOVyDegrees
        }
        return ScenePose(eye: eye, yaw: yaw, pitch: pitch, fovyDegrees: fovy)
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

    /// Whether this COLMAP build offers `subcommand`. A valid subcommand's --help
    /// lists `--database_path`; an unknown one prints an error without it.
    private func colmapSupports(_ subcommand: String) -> Bool {
        helpText(colmap, subcommand).contains("database_path")
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
                         env: [String: String]? = nil, onLine: ((String) -> Void)? = nil) throws {
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
        if let env {
            process.environment = ProcessInfo.processInfo.environment.merging(env) { _, new in new }
        }

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
