#!/usr/bin/env swift
// Render an SF Symbol over a gradient into a complete macOS .iconset of PNGs.
// Then bundle them into AppIcon.icns:
//   swift Resources/make_icon.swift
//   iconutil -c icns Resources/AppIcon.iconset -o Resources/AppIcon.icns
//   rm -rf Resources/AppIcon.iconset
// (scripts/build-app.sh copies Resources/AppIcon.icns into the app bundle.)

import AppKit
import Foundation

let symbolName = "cube.transparent"
let topColor = NSColor(calibratedRed: 0.42, green: 0.34, blue: 0.95, alpha: 1)   // indigo
let bottomColor = NSColor(calibratedRed: 0.80, green: 0.31, blue: 0.86, alpha: 1) // violet
let foregroundColor = NSColor.white

let scriptDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let iconsetURL = scriptDir.appendingPathComponent("AppIcon.iconset")
let preview1024URL = scriptDir.appendingPathComponent("AppIcon.png")

let sizes: [(point: Int, scale: Int)] = [
    (16, 1), (16, 2),
    (32, 1), (32, 2),
    (128, 1), (128, 2),
    (256, 1), (256, 2),
    (512, 1), (512, 2),
]

func renderPNG(pixelSize: Int) -> Data? {
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: pixelSize, pixelsHigh: pixelSize,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
    ) else { return nil }
    rep.size = NSSize(width: pixelSize, height: pixelSize)

    NSGraphicsContext.saveGraphicsState()
    defer { NSGraphicsContext.restoreGraphicsState() }
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

    let size = CGFloat(pixelSize)
    let cornerRadius = size * 0.225
    let bg = NSBezierPath(
        roundedRect: NSRect(x: 0, y: 0, width: size, height: size),
        xRadius: cornerRadius, yRadius: cornerRadius
    )
    if let gradient = NSGradient(starting: topColor, ending: bottomColor) {
        gradient.draw(in: bg, angle: -90)
    } else {
        topColor.setFill(); bg.fill()
    }

    guard let symbol = NSImage(systemSymbolName: symbolName, accessibilityDescription: nil) else {
        return rep.representation(using: .png, properties: [:])
    }
    let config = NSImage.SymbolConfiguration(pointSize: size * 0.56, weight: .semibold)
        .applying(NSImage.SymbolConfiguration(paletteColors: [foregroundColor]))
    let configured = symbol.withSymbolConfiguration(config) ?? symbol
    let symbolSize = configured.size
    let symbolRect = NSRect(
        x: (size - symbolSize.width) / 2,
        y: (size - symbolSize.height) / 2,
        width: symbolSize.width, height: symbolSize.height
    )
    configured.draw(in: symbolRect)

    return rep.representation(using: .png, properties: [:])
}

let fm = FileManager.default
try? fm.removeItem(at: iconsetURL)
try fm.createDirectory(at: iconsetURL, withIntermediateDirectories: true)

for (pt, scale) in sizes {
    let px = pt * scale
    guard let data = renderPNG(pixelSize: px) else {
        FileHandle.standardError.write(Data("failed to render \(px)x\(px)\n".utf8)); exit(1)
    }
    let suffix = scale == 1 ? "" : "@2x"
    try data.write(to: iconsetURL.appendingPathComponent("icon_\(pt)x\(pt)\(suffix).png"))
}

if let preview = renderPNG(pixelSize: 1024) {
    try preview.write(to: preview1024URL)
}

print("wrote \(iconsetURL.path) and \(preview1024URL.path)")
