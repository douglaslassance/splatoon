import Foundation
import CoreML
import simd

// MARK: - PLY -> Gaussians
//
// Reads a Gaussian-splat PLY back into a `Gaussians3D` so the mesh exporter can
// rebuild from a cached splat without re-running inference. Properties are keyed
// by *name*, so this accepts both:
//   - SHARP splats written by `SplatExporter` (14 props: x,y,z, f_dc_0..2,
//     opacity, scale_0..2, rot_0..3), and
//   - standard 3DGS splats from COLMAP + a trainer (adds nx/ny/nz normals and
//     f_rest_* SH bands, which are simply ignored here).
// The per-attribute decoding (SH degree-0 -> colour, logit -> opacity,
// log -> scale) is identical for both, since both use the INRIA conventions.

/// Sigmoid, the inverse of `inverseSigmoid`.
private func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }

/// sRGB -> linear RGB, the inverse of `linearRGBToSRGB`.
private func srgbToLinearRGB(_ c: Float) -> Float {
    c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
}

/// Degree-0 spherical-harmonics coefficient -> RGB.
private func sphericalHarmonicsToRGB(_ sh: Float) -> Float {
    let coeffDegree0 = sqrt(1.0 / (4.0 * Float.pi))
    return sh * coeffDegree0 + 0.5
}

enum SplatPLYReader {

    enum ReadError: LocalizedError {
        case badHeader
        case unsupportedFormat(String)
        case missingProperties([String])
        case truncated

        var errorDescription: String? {
            switch self {
            case .badHeader: return "Not a readable splat PLY (missing header)."
            case .unsupportedFormat(let f): return "Unsupported PLY format: \(f)."
            case .missingProperties(let names):
                return "Splat PLY is missing expected properties: \(names.joined(separator: ", "))."
            case .truncated: return "Splat PLY is truncated."
            }
        }
    }

    /// Ordered vertex property with its scalar type and byte size.
    private struct Property { let name: String; let type: String; let size: Int }

    private static func typeSize(_ t: String) -> Int {
        switch t {
        case "char", "uchar", "int8", "uint8": return 1
        case "short", "ushort", "int16", "uint16": return 2
        case "int", "uint", "int32", "uint32", "float", "float32": return 4
        case "double", "float64", "int64", "uint64": return 8
        default: return 4
        }
    }

    /// Load a Gaussian-splat PLY (SHARP or standard 3DGS) into a `Gaussians3D`.
    static func readGaussians(from url: URL) throws -> Gaussians3D {
        let data = try Data(contentsOf: url)

        guard let headerRange = data.range(of: Data("end_header\n".utf8)) else {
            throw ReadError.badHeader
        }
        guard let header = String(data: data.subdata(in: 0..<headerRange.upperBound), encoding: .ascii) else {
            throw ReadError.badHeader
        }

        // ===== Parse the header: format + the vertex element's properties =====
        var format = "binary_little_endian"
        var vertexCount = 0
        var properties: [Property] = []
        var inVertexElement = false
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ").map(String.init)
            guard let keyword = parts.first else { continue }
            switch keyword {
            case "format" where parts.count >= 2:
                format = parts[1]
            case "element" where parts.count >= 3:
                inVertexElement = (parts[1] == "vertex")
                if inVertexElement { vertexCount = Int(parts[2]) ?? 0 }
            case "property" where inVertexElement && parts.count >= 3:
                // Skip list properties (faces etc.); vertex attributes are scalars.
                if parts[1] == "list" { continue }
                properties.append(Property(name: parts[2], type: parts[1], size: typeSize(parts[1])))
            default:
                break
            }
        }
        guard vertexCount > 0, !properties.isEmpty else { throw ReadError.badHeader }

        // Offsets of every property within one vertex record.
        var offsetOf: [String: (offset: Int, type: String)] = [:]
        var cursor = 0
        for p in properties {
            offsetOf[p.name] = (cursor, p.type)
            cursor += p.size
        }
        let stride = cursor

        let needed = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                      "opacity", "scale_0", "scale_1", "scale_2",
                      "rot_0", "rot_1", "rot_2", "rot_3"]
        let missing = needed.filter { offsetOf[$0] == nil }
        guard missing.isEmpty else { throw ReadError.missingProperties(missing) }

        let n = vertexCount
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

        switch format {
        case "binary_little_endian":
            try readBinaryLE(data: data, bodyStart: headerRange.upperBound, n: n, stride: stride,
                             offsetOf: offsetOf, meanPtr: meanPtr, scalePtr: scalePtr,
                             quatPtr: quatPtr, colorPtr: colorPtr, opacityPtr: opacityPtr)
        case "ascii":
            try readASCII(data: data, bodyStart: headerRange.upperBound, n: n,
                          propertyOrder: properties.map(\.name),
                          meanPtr: meanPtr, scalePtr: scalePtr, quatPtr: quatPtr,
                          colorPtr: colorPtr, opacityPtr: opacityPtr)
        default:
            throw ReadError.unsupportedFormat(format)
        }

        return Gaussians3D(meanVectors: mean, singularValues: scale,
                           quaternions: quat, colors: color, opacities: opacity)
    }

    // MARK: - Decode one vertex's named values into the output arrays

    private static func store(i: Int,
                              value: (String) -> Float,
                              meanPtr: UnsafeMutablePointer<Float>,
                              scalePtr: UnsafeMutablePointer<Float>,
                              quatPtr: UnsafeMutablePointer<Float>,
                              colorPtr: UnsafeMutablePointer<Float>,
                              opacityPtr: UnsafeMutablePointer<Float>) {
        meanPtr[i * 3 + 0] = value("x")
        meanPtr[i * 3 + 1] = value("y")
        meanPtr[i * 3 + 2] = value("z")

        // Colour: SH degree-0 -> sRGB -> linear (the mesh exporter re-applies sRGB).
        colorPtr[i * 3 + 0] = srgbToLinearRGB(sphericalHarmonicsToRGB(value("f_dc_0")))
        colorPtr[i * 3 + 1] = srgbToLinearRGB(sphericalHarmonicsToRGB(value("f_dc_1")))
        colorPtr[i * 3 + 2] = srgbToLinearRGB(sphericalHarmonicsToRGB(value("f_dc_2")))

        opacityPtr[i] = sigmoid(value("opacity"))

        scalePtr[i * 3 + 0] = exp(value("scale_0"))
        scalePtr[i * 3 + 1] = exp(value("scale_1"))
        scalePtr[i * 3 + 2] = exp(value("scale_2"))

        // Normalize the quaternion (w,x,y,z): SHARP's are already unit, but a
        // trainer may store raw values, and the mesh maths assumes unit length.
        var q = SIMD4<Float>(value("rot_0"), value("rot_1"), value("rot_2"), value("rot_3"))
        let len = simd_length(q)
        if len > 1e-12 { q /= len } else { q = SIMD4(1, 0, 0, 0) }
        quatPtr[i * 4 + 0] = q.x
        quatPtr[i * 4 + 1] = q.y
        quatPtr[i * 4 + 2] = q.z
        quatPtr[i * 4 + 3] = q.w
    }

    private static func readBinaryLE(data: Data, bodyStart: Int, n: Int, stride: Int,
                                     offsetOf: [String: (offset: Int, type: String)],
                                     meanPtr: UnsafeMutablePointer<Float>,
                                     scalePtr: UnsafeMutablePointer<Float>,
                                     quatPtr: UnsafeMutablePointer<Float>,
                                     colorPtr: UnsafeMutablePointer<Float>,
                                     opacityPtr: UnsafeMutablePointer<Float>) throws {
        guard bodyStart + n * stride <= data.count else { throw ReadError.truncated }
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            func read(_ base: Int, _ prop: (offset: Int, type: String)) -> Float {
                let off = base + prop.offset
                switch prop.type {
                case "double", "float64": return Float(raw.loadUnaligned(fromByteOffset: off, as: Float64.self))
                case "uchar", "uint8":    return Float(raw.loadUnaligned(fromByteOffset: off, as: UInt8.self))
                case "char", "int8":      return Float(raw.loadUnaligned(fromByteOffset: off, as: Int8.self))
                case "ushort", "uint16":  return Float(raw.loadUnaligned(fromByteOffset: off, as: UInt16.self))
                case "short", "int16":    return Float(raw.loadUnaligned(fromByteOffset: off, as: Int16.self))
                case "uint", "uint32":    return Float(raw.loadUnaligned(fromByteOffset: off, as: UInt32.self))
                case "int", "int32":      return Float(raw.loadUnaligned(fromByteOffset: off, as: Int32.self))
                default:                  return raw.loadUnaligned(fromByteOffset: off, as: Float32.self)
                }
            }
            for i in 0..<n {
                let base = bodyStart + i * stride
                store(i: i, value: { read(base, offsetOf[$0]!) },
                      meanPtr: meanPtr, scalePtr: scalePtr, quatPtr: quatPtr,
                      colorPtr: colorPtr, opacityPtr: opacityPtr)
            }
        }
    }

    private static func readASCII(data: Data, bodyStart: Int, n: Int, propertyOrder: [String],
                                  meanPtr: UnsafeMutablePointer<Float>,
                                  scalePtr: UnsafeMutablePointer<Float>,
                                  quatPtr: UnsafeMutablePointer<Float>,
                                  colorPtr: UnsafeMutablePointer<Float>,
                                  opacityPtr: UnsafeMutablePointer<Float>) throws {
        guard let body = String(data: data.subdata(in: bodyStart..<data.count), encoding: .ascii) else {
            throw ReadError.truncated
        }
        var indexOf: [String: Int] = [:]
        for (i, name) in propertyOrder.enumerated() { indexOf[name] = i }

        let rows = body.split(whereSeparator: \.isNewline)
        guard rows.count >= n else { throw ReadError.truncated }
        for i in 0..<n {
            let tokens = rows[i].split(whereSeparator: \.isWhitespace).map { Float($0) ?? 0 }
            store(i: i, value: { tokens[indexOf[$0] ?? 0] },
                  meanPtr: meanPtr, scalePtr: scalePtr, quatPtr: quatPtr,
                  colorPtr: colorPtr, opacityPtr: opacityPtr)
        }
    }
}
