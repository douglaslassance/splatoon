import Foundation
import CoreML

// MARK: - PLY -> Gaussians (inverse of SplatExporter.savePLY)
//
// Reads a splat PLY written by `SplatExporter` back into a `Gaussians3D`, undoing
// the encodings (spherical harmonics -> colour, logit -> opacity, log -> scale).
// This lets the mesh exporter rebuild from a cached splat without re-running
// inference, so meshing can be iterated on independently.

/// Sigmoid, the inverse of `inverseSigmoid`.
private func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }

/// sRGB -> linear RGB, the inverse of `linearRGBToSRGB`.
private func srgbToLinearRGB(_ c: Float) -> Float {
    c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
}

/// Degree-0 spherical-harmonics coefficient -> RGB, the inverse of
/// `rgbToSphericalHarmonics`.
private func sphericalHarmonicsToRGB(_ sh: Float) -> Float {
    let coeffDegree0 = sqrt(1.0 / (4.0 * Float.pi))
    return sh * coeffDegree0 + 0.5
}

enum SplatPLYReader {

    enum ReadError: LocalizedError {
        case badHeader
        case unexpectedLayout(Int)
        case truncated

        var errorDescription: String? {
            switch self {
            case .badHeader: return "Not a readable splat PLY (missing header)."
            case .unexpectedLayout(let count):
                return "Unexpected vertex layout (\(count) properties); expected 14."
            case .truncated: return "Splat PLY is truncated."
            }
        }
    }

    /// Load the vertex block of a `SplatExporter` PLY into a `Gaussians3D`.
    static func readGaussians(from url: URL) throws -> Gaussians3D {
        let data = try Data(contentsOf: url)

        // ===== Parse the ASCII header =====
        guard let headerRange = data.range(of: Data("end_header\n".utf8)) else {
            throw ReadError.badHeader
        }
        let headerData = data.subdata(in: 0..<headerRange.upperBound)
        guard let header = String(data: headerData, encoding: .ascii) else {
            throw ReadError.badHeader
        }

        var vertexCount = 0
        var vertexProps = 0
        var seenVertexElement = false
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ")
            if parts.count >= 3, parts[0] == "element" {
                seenVertexElement = (parts[1] == "vertex")
                if seenVertexElement { vertexCount = Int(parts[2]) ?? 0 }
            } else if parts.count >= 2, parts[0] == "property", seenVertexElement {
                vertexProps += 1
            }
        }
        // Our writer emits exactly: x,y,z, f_dc_0..2, opacity, scale_0..2, rot_0..3.
        guard vertexProps == 14 else { throw ReadError.unexpectedLayout(vertexProps) }
        guard vertexCount > 0 else { throw ReadError.badHeader }

        let n = vertexCount
        let floatsPerVertex = 14
        let byteCount = n * floatsPerVertex * MemoryLayout<Float>.size
        let start = headerRange.upperBound
        guard start + byteCount <= data.count else { throw ReadError.truncated }

        // ===== Read the interleaved vertex floats =====
        let floats: [Float] = data.subdata(in: start..<(start + byteCount)).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }

        let mean = try MLMultiArray(shape: [1, NSNumber(value: n), 3], dataType: .float32)
        let scale = try MLMultiArray(shape: [1, NSNumber(value: n), 3], dataType: .float32)
        let quat = try MLMultiArray(shape: [1, NSNumber(value: n), 4], dataType: .float32)
        let color = try MLMultiArray(shape: [1, NSNumber(value: n), 3], dataType: .float32)
        let opacity = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .float32)

        let meanPtr = mean.dataPointer.assumingMemoryBound(to: Float.self)
        let scalePtr = scale.dataPointer.assumingMemoryBound(to: Float.self)
        let quatPtr = quat.dataPointer.assumingMemoryBound(to: Float.self)
        let colorPtr = color.dataPointer.assumingMemoryBound(to: Float.self)
        let opacityPtr = opacity.dataPointer.assumingMemoryBound(to: Float.self)

        for i in 0..<n {
            let base = i * floatsPerVertex
            meanPtr[i * 3 + 0] = floats[base + 0]
            meanPtr[i * 3 + 1] = floats[base + 1]
            meanPtr[i * 3 + 2] = floats[base + 2]

            // Colour: SH -> sRGB -> linear (the mesh exporter re-applies sRGB).
            colorPtr[i * 3 + 0] = srgbToLinearRGB(sphericalHarmonicsToRGB(floats[base + 3]))
            colorPtr[i * 3 + 1] = srgbToLinearRGB(sphericalHarmonicsToRGB(floats[base + 4]))
            colorPtr[i * 3 + 2] = srgbToLinearRGB(sphericalHarmonicsToRGB(floats[base + 5]))

            opacityPtr[i] = sigmoid(floats[base + 6])

            scalePtr[i * 3 + 0] = exp(floats[base + 7])
            scalePtr[i * 3 + 1] = exp(floats[base + 8])
            scalePtr[i * 3 + 2] = exp(floats[base + 9])

            quatPtr[i * 4 + 0] = floats[base + 10]
            quatPtr[i * 4 + 1] = floats[base + 11]
            quatPtr[i * 4 + 2] = floats[base + 12]
            quatPtr[i * 4 + 3] = floats[base + 13]
        }

        return Gaussians3D(meanVectors: mean, singularValues: scale,
                           quaternions: quat, colors: color, opacities: opacity)
    }
}
