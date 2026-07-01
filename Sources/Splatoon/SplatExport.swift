import Foundation
import CoreML

// MARK: - Color / activation utilities
//
// Ported verbatim from the community reference runner (pearsonkyle/Sharp-coreml,
// sharp.swift) so the exported PLY matches the validated numerics exactly.

/// Convert linear RGB to sRGB color space.
func linearRGBToSRGB(_ linear: Float) -> Float {
    if linear <= 0.0031308 {
        return linear * 12.92
    } else {
        return 1.055 * pow(linear, 1.0 / 2.4) - 0.055
    }
}

/// Convert an RGB value to degree-0 spherical-harmonics coefficient.
func rgbToSphericalHarmonics(_ rgb: Float) -> Float {
    let coeffDegree0 = sqrt(1.0 / (4.0 * Float.pi))
    return (rgb - 0.5) / coeffDegree0
}

/// Inverse sigmoid (logit), matching the Python `save_ply` opacity encoding.
func inverseSigmoid(_ x: Float) -> Float {
    let clamped = min(max(x, 1e-6), 1.0 - 1e-6)
    return log(clamped / (1.0 - clamped))
}

// MARK: - PLY export

enum SplatExporter {

    /// Save Gaussians to a binary PLY file, matching the reference runner's
    /// `save_ply` format exactly (compatible with MetalSplatter / Splat Viewer).
    ///
    /// - Parameters:
    ///   - gaussians: model output.
    ///   - focalLengthPx: focal length in pixels (used for the intrinsic matrix).
    ///   - imageShape: dimensions the model ran at (height, width).
    ///   - outputURL: destination file.
    ///   - decimation: 0.0–1.0. 1.0 keeps all Gaussians; 0.5 keeps the 50% most important.
    static func savePLY(gaussians: Gaussians3D,
                        focalLengthPx: Float,
                        imageShape: (height: Int, width: Int),
                        to outputURL: URL,
                        decimation: Float = 1.0) throws {

        let imageHeight = imageShape.height
        let imageWidth = imageShape.width

        let originalCount = gaussians.count
        let keepIndices: [Int]
        if decimation < 1.0 {
            keepIndices = gaussians.decimationIndices(keepRatio: decimation)
        } else {
            keepIndices = Array(0..<originalCount)
        }
        let numGaussians = keepIndices.count

        var fileContent = Data()

        func appendString(_ str: String) {
            fileContent.append(str.data(using: .ascii)!)
        }
        func appendFloat32(_ value: Float) {
            var v = value
            fileContent.append(Data(bytes: &v, count: 4))
        }
        func appendInt32(_ value: Int32) {
            var v = value
            fileContent.append(Data(bytes: &v, count: 4))
        }
        func appendUInt32(_ value: UInt32) {
            var v = value
            fileContent.append(Data(bytes: &v, count: 4))
        }
        func appendUInt8(_ value: UInt8) {
            var v = value
            fileContent.append(Data(bytes: &v, count: 1))
        }

        // ===== Header =====
        appendString("ply\n")
        appendString("format binary_little_endian 1.0\n")
        appendString("element vertex \(numGaussians)\n")
        for prop in ["x", "y", "z",
                     "f_dc_0", "f_dc_1", "f_dc_2",
                     "opacity",
                     "scale_0", "scale_1", "scale_2",
                     "rot_0", "rot_1", "rot_2", "rot_3"] {
            appendString("property float \(prop)\n")
        }
        appendString("element extrinsic 16\n")
        appendString("property float extrinsic\n")
        appendString("element intrinsic 9\n")
        appendString("property float intrinsic\n")
        appendString("element image_size 2\n")
        appendString("property uint image_size\n")
        appendString("element frame 2\n")
        appendString("property int frame\n")
        appendString("element disparity 2\n")
        appendString("property float disparity\n")
        appendString("element color_space 1\n")
        appendString("property uchar color_space\n")
        appendString("element version 3\n")
        appendString("property uchar version\n")
        appendString("end_header\n")

        // ===== Vertex data =====
        var disparities: [Float] = []

        let meanPtr = gaussians.meanVectors.dataPointer.assumingMemoryBound(to: Float.self)
        let scalePtr = gaussians.singularValues.dataPointer.assumingMemoryBound(to: Float.self)
        let quatPtr = gaussians.quaternions.dataPointer.assumingMemoryBound(to: Float.self)
        let colorPtr = gaussians.colors.dataPointer.assumingMemoryBound(to: Float.self)
        let opacityPtr = gaussians.opacities.dataPointer.assumingMemoryBound(to: Float.self)

        for i in keepIndices {
            let x = meanPtr[i * 3 + 0]
            let y = meanPtr[i * 3 + 1]
            let z = meanPtr[i * 3 + 2]
            appendFloat32(x)
            appendFloat32(y)
            appendFloat32(z)

            if z > 1e-6 {
                disparities.append(1.0 / z)
            }

            // Colors: linear RGB -> sRGB -> degree-0 spherical harmonics.
            let srgbR = linearRGBToSRGB(colorPtr[i * 3 + 0])
            let srgbG = linearRGBToSRGB(colorPtr[i * 3 + 1])
            let srgbB = linearRGBToSRGB(colorPtr[i * 3 + 2])
            appendFloat32(rgbToSphericalHarmonics(srgbR))
            appendFloat32(rgbToSphericalHarmonics(srgbG))
            appendFloat32(rgbToSphericalHarmonics(srgbB))

            // Opacity -> logit.
            appendFloat32(inverseSigmoid(opacityPtr[i]))

            // Scales -> log space.
            appendFloat32(log(max(scalePtr[i * 3 + 0], 1e-10)))
            appendFloat32(log(max(scalePtr[i * 3 + 1], 1e-10)))
            appendFloat32(log(max(scalePtr[i * 3 + 2], 1e-10)))

            // Quaternions (w, x, y, z) passed through.
            appendFloat32(quatPtr[i * 4 + 0])
            appendFloat32(quatPtr[i * 4 + 1])
            appendFloat32(quatPtr[i * 4 + 2])
            appendFloat32(quatPtr[i * 4 + 3])
        }

        // ===== Extrinsic (4x4 identity) =====
        for val in [Float(1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] {
            appendFloat32(val)
        }

        // ===== Intrinsic (3x3) =====
        let intrinsic: [Float] = [
            focalLengthPx, 0, Float(imageWidth) * 0.5,
            0, focalLengthPx, Float(imageHeight) * 0.5,
            0, 0, 1
        ]
        for val in intrinsic { appendFloat32(val) }

        // ===== Image size =====
        appendUInt32(UInt32(imageWidth))
        appendUInt32(UInt32(imageHeight))

        // ===== Frame =====
        appendInt32(1)                     // number of frames
        appendInt32(Int32(numGaussians))   // particles per frame

        // ===== Disparity quantiles =====
        disparities.sort()
        let q10Index = Int(Float(disparities.count) * 0.1)
        let q90Index = Int(Float(disparities.count) * 0.9)
        let disparity10 = disparities.isEmpty ? 0.0 : disparities[min(q10Index, disparities.count - 1)]
        let disparity90 = disparities.isEmpty ? 1.0 : disparities[min(q90Index, disparities.count - 1)]
        appendFloat32(disparity10)
        appendFloat32(disparity90)

        // ===== Color space (sRGB = 1) =====
        appendUInt8(1)

        // ===== Version (1.5.0) =====
        appendUInt8(1)
        appendUInt8(5)
        appendUInt8(0)

        try fileContent.write(to: outputURL)
    }
}
