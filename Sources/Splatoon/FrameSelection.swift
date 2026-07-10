import CoreGraphics

/// Pure image-quality scoring used to cull junk frames before they reach COLMAP.
/// Blurred, near-black, or blown-out frames waste matching time and drag down the
/// reconstruction, so they're dropped up front. All scoring runs on a small
/// grayscale downscale, so scores are comparable across sources regardless of
/// original resolution.
enum FrameSelection {

    /// Exposure below/above these fractions carries no usable features (near-black
    /// or blown-out); reject outright.
    static let exposureFloor = 0.03
    static let exposureCeil = 0.97
    /// Keep only frames at least this fraction of a video's median sharpness (a
    /// scale-free floor that drops the blurriest frames without a magic absolute
    /// threshold).
    static let relativeSharpnessFloor = 0.5

    /// ~3 fps video sampling, plus per-video and total frame budgets. Caps keep
    /// COLMAP's O(n²) matching tractable while leaving enough overlap to register.
    static let sampleFPS = 3.0
    static let maxVideoSamples = 72       // oversampling ceiling for scoring
    static let minFramesPerVideo = 15
    static let maxFramesPerVideo = 48
    static let totalFrameBudget = 120

    /// Working size for the downscale everything is scored on.
    private static let workingSize = 256

    /// Sharpness = variance of the Laplacian over the grayscale downscale. High
    /// for crisp frames, near-zero for blurred or flat ones.
    static func sharpness(of image: CGImage) -> Double {
        guard let (px, w, h) = grayscale(image), w > 2, h > 2 else { return 0 }
        var sum = 0.0, sumSq = 0.0, n = 0.0
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                let lap = -4.0 * Double(px[i])
                    + Double(px[i - 1]) + Double(px[i + 1])
                    + Double(px[i - w]) + Double(px[i + w])
                sum += lap
                sumSq += lap * lap
                n += 1
            }
        }
        guard n > 0 else { return 0 }
        let mean = sum / n
        return max(0, sumSq / n - mean * mean)
    }

    /// Mean luminance, 0…1, over the grayscale downscale.
    static func exposure(of image: CGImage) -> Double {
        guard let (px, _, _) = grayscale(image), !px.isEmpty else { return 0 }
        var sum = 0.0
        for v in px { sum += Double(v) }
        return sum / Double(px.count) / 255.0
    }

    /// Reject frames whose exposure carries no usable signal (near-black/blown).
    /// Sharpness culling is done relatively, per source, so it's not gated here.
    static func exposureUsable(_ exposure: Double) -> Bool {
        exposure > exposureFloor && exposure < exposureCeil
    }

    /// Draw `image` into a small 8-bit grayscale buffer for scoring.
    private static func grayscale(_ image: CGImage) -> (pixels: [UInt8], width: Int, height: Int)? {
        let longest = max(image.width, image.height)
        guard longest > 0 else { return nil }
        let scale = Double(workingSize) / Double(longest)
        let w = max(1, Int((Double(image.width) * scale).rounded()))
        let h = max(1, Int((Double(image.height) * scale).rounded()))
        var pixels = [UInt8](repeating: 0, count: w * h)
        let ok: Bool = pixels.withUnsafeMutableBytes { buffer in
            guard let ctx = CGContext(data: buffer.baseAddress, width: w, height: h,
                                      bitsPerComponent: 8, bytesPerRow: w,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? (pixels, w, h) : nil
    }
}
