// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Splatoon",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/scier/MetalSplatter", exact: "1.0.1")
    ],
    targets: [
        .executableTarget(
            name: "Splatoon",
            dependencies: [
                .product(name: "MetalSplatter", package: "MetalSplatter"),
                .product(name: "SplatIO", package: "MetalSplatter"),
                .product(name: "PLYIO", package: "MetalSplatter")
            ],
            path: "Sources/Splatoon",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
