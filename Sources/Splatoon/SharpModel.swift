import Foundation
import CoreML
import CoreImage

/// The SHARP model output — a collection of 3D Gaussians.
/// Ported from the community reference runner (pearsonkyle/Sharp-coreml).
struct Gaussians3D {
    let meanVectors: MLMultiArray      // (1, N, 3) positions
    let singularValues: MLMultiArray   // (1, N, 3) scales
    let quaternions: MLMultiArray      // (1, N, 4) rotations (w, x, y, z)
    let colors: MLMultiArray           // (1, N, 3) linear RGB
    let opacities: MLMultiArray        // (1, N) opacity

    var count: Int { meanVectors.shape[1].intValue }

    /// Importance = product of scales × opacity. Higher = keep during decimation.
    func computeImportanceScores() -> [Float] {
        let n = count
        var scores = [Float](repeating: 0, count: n)
        let scalePtr = singularValues.dataPointer.assumingMemoryBound(to: Float.self)
        let opacityPtr = opacities.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<n {
            let s0 = scalePtr[i * 3 + 0]
            let s1 = scalePtr[i * 3 + 1]
            let s2 = scalePtr[i * 3 + 2]
            scores[i] = (s0 * s1 * s2) * opacityPtr[i]
        }
        return scores
    }

    /// Indices of the most important Gaussians to keep, sorted ascending for
    /// spatial coherence.
    func decimationIndices(keepRatio: Float) -> [Int] {
        let n = count
        let keepCount = max(1, Int(Float(n) * keepRatio))
        let scores = computeImportanceScores()
        var indexed = scores.enumerated().map { ($0.offset, $0.element) }
        indexed.sort { $0.1 > $1.1 }
        var keep = indexed.prefix(keepCount).map { $0.0 }
        keep.sort()
        return keep
    }
}

/// Loads a SHARP Core ML model and runs single-image inference.
final class SharpModelRunner {
    private let model: MLModel
    let inputHeight: Int
    let inputWidth: Int

    init(modelURL: URL, inputHeight: Int = 1536, inputWidth: Int = 1536) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let compiled = try SharpModelRunner.compileModelIfNeeded(at: modelURL)
        self.model = try MLModel(contentsOf: compiled, configuration: config)
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
    }

    // MARK: - Compilation cache

    private static func compileModelIfNeeded(at modelURL: URL) throws -> URL {
        let fm = FileManager.default
        let ext = modelURL.pathExtension.lowercased()

        if ext == "mlmodelc" { return modelURL }
        guard ext == "mlpackage" || ext == "mlmodel" else {
            throw SharpError.unsupportedModel(ext)
        }

        let cacheDir = fm.temporaryDirectory.appendingPathComponent("SHARPModelCache")
        try? fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let modelName = modelURL.deletingPathExtension().lastPathComponent
        let compiledPath = cacheDir.appendingPathComponent("\(modelName).mlmodelc")

        if fm.fileExists(atPath: compiledPath.path) {
            let sourceAttrs = try fm.attributesOfItem(atPath: modelURL.path)
            let cachedAttrs = try fm.attributesOfItem(atPath: compiledPath.path)
            if let sourceDate = sourceAttrs[.modificationDate] as? Date,
               let cachedDate = cachedAttrs[.modificationDate] as? Date,
               cachedDate >= sourceDate {
                return compiledPath
            }
            try? fm.removeItem(at: compiledPath)
        }

        let temporaryCompiledURL = try MLModel.compileModel(at: modelURL)
        try? fm.removeItem(at: compiledPath)
        try fm.moveItem(at: temporaryCompiledURL, to: compiledPath)
        return compiledPath
    }

    // MARK: - Preprocessing

    /// Resize a `CGImage` to the model's input resolution and pack it into a
    /// `(1, 3, H, W)` float32 `MLMultiArray` normalized to [0, 1].
    func preprocess(_ cgImage: CGImage) throws -> MLMultiArray {
        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext()

        let scaleX = CGFloat(inputWidth) / ciImage.extent.width
        let scaleY = CGFloat(inputHeight) / ciImage.extent.height
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        guard let resized = context.createCGImage(
            scaled,
            from: CGRect(x: 0, y: 0, width: inputWidth, height: inputHeight)
        ) else {
            throw SharpError.preprocessingFailed
        }

        let imageArray = try MLMultiArray(
            shape: [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)],
            dataType: .float32
        )

        let width = resized.width
        let height = resized.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let cgContext = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw SharpError.preprocessingFailed
        }
        cgContext.draw(resized, in: CGRect(x: 0, y: 0, width: width, height: height))

        let ptr = imageArray.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = inputHeight * inputWidth
        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * bytesPerPixel
                let spatialIndex = y * inputWidth + x
                ptr[0 * channelStride + spatialIndex] = Float(pixelData[pixelIndex]) / 255.0
                ptr[1 * channelStride + spatialIndex] = Float(pixelData[pixelIndex + 1]) / 255.0
                ptr[2 * channelStride + spatialIndex] = Float(pixelData[pixelIndex + 2]) / 255.0
            }
        }
        return imageArray
    }

    // MARK: - Inference

    func predict(image: MLMultiArray, focalLengthPx: Float) throws -> Gaussians3D {
        let disparityFactor = focalLengthPx / Float(inputWidth)
        let disparityArray = try MLMultiArray(shape: [1], dataType: .float32)
        disparityArray[0] = NSNumber(value: disparityFactor)

        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: image),
            "disparity_factor": MLFeatureValue(multiArray: disparityArray)
        ])

        let output = try model.prediction(from: inputFeatures)
        let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)

        func findOutput(containing keywords: [String]) -> MLMultiArray? {
            for name in outputNames {
                let lower = name.lowercased()
                for keyword in keywords where lower.contains(keyword.lowercased()) {
                    return output.featureValue(for: name)?.multiArrayValue
                }
            }
            return nil
        }

        let meanVectors = output.featureValue(for: "mean_vectors_3d_positions")?.multiArrayValue
            ?? findOutput(containing: ["mean", "position", "xyz"])
        let singularValues = output.featureValue(for: "singular_values_scales")?.multiArrayValue
            ?? findOutput(containing: ["singular", "scale"])
        let quaternions = output.featureValue(for: "quaternions_rotations")?.multiArrayValue
            ?? findOutput(containing: ["quaternion", "rotation", "rot"])
        let colors = output.featureValue(for: "colors_rgb_linear")?.multiArrayValue
            ?? findOutput(containing: ["color", "rgb"])
        let opacities = output.featureValue(for: "opacities_alpha_channel")?.multiArrayValue
            ?? findOutput(containing: ["opacity", "alpha"])

        guard let mv = meanVectors, let sv = singularValues, let q = quaternions,
              let c = colors, let o = opacities else {
            throw SharpError.missingOutputs(outputNames)
        }

        return Gaussians3D(meanVectors: mv, singularValues: sv,
                           quaternions: q, colors: c, opacities: o)
    }
}

enum SharpError: LocalizedError {
    case unsupportedModel(String)
    case preprocessingFailed
    case missingOutputs([String])

    var errorDescription: String? {
        switch self {
        case .unsupportedModel(let ext):
            return "Unsupported model format “.\(ext)”. Use .mlpackage, .mlmodel, or .mlmodelc."
        case .preprocessingFailed:
            return "Failed to prepare the image for the model."
        case .missingOutputs(let names):
            return "The model did not return the expected outputs. Got: \(names.joined(separator: ", "))."
        }
    }
}
