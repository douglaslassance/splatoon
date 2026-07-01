#!/usr/bin/env bash
#
# Builds Splatoon.app from the SwiftPM package — no Xcode project required.
# Produces ./Splatoon.app, ad-hoc code-signed and ready to `open`.
#
# Usage:
#   scripts/build-app.sh [debug|release]   (default: release)

set -euo pipefail

CONFIG="${1:-release}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

APP="$ROOT/Splatoon.app"
BUNDLE_ID="com.douglaslassance.Splatoon"
EXECUTABLE="Splatoon"

echo "==> swift build -c $CONFIG"
swift build -c "$CONFIG"

BIN_DIR="$(swift build -c "$CONFIG" --show-bin-path)"

echo "==> Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BIN_DIR/$EXECUTABLE" "$APP/Contents/MacOS/$EXECUTABLE"

# Copy SwiftPM resource bundles (MetalSplatter ships its shaders in a
# *_MetalSplatter.bundle). Bundle.module resolves these from Resources.
shopt -s nullglob
for bundle in "$BIN_DIR"/*.bundle; do
  cp -R "$bundle" "$APP/Contents/Resources/"
done
shopt -u nullglob

# `swift build` copies MetalSplatter's .metal files as source but does NOT
# compile them (only Xcode's build system does). SplatRenderer loads a compiled
# `default.metallib` via makeDefaultLibrary(bundle:), so compile it ourselves.
METAL_RES="$ROOT/.build/checkouts/MetalSplatter/MetalSplatter/Resources"
DEST_BUNDLE="$APP/Contents/Resources/MetalSplatter_MetalSplatter.bundle"
if [ -d "$METAL_RES" ] && [ -d "$DEST_BUNDLE" ]; then
  echo "==> Compiling Metal shaders -> default.metallib"
  TMP_AIR="$(mktemp -d)"
  for f in "$METAL_RES"/*.metal; do
    xcrun -sdk macosx metal -I "$METAL_RES" -c "$f" \
      -o "$TMP_AIR/$(basename "${f%.metal}").air"
  done
  xcrun -sdk macosx metallib "$TMP_AIR"/*.air -o "$DEST_BUNDLE/default.metallib"
  rm -rf "$TMP_AIR"
else
  echo "warning: Metal resources not found; viewer shaders will be missing" >&2
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Splatoon</string>
    <key>CFBundleDisplayName</key><string>Splatoon</string>
    <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
    <key>CFBundleExecutable</key><string>$EXECUTABLE</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>15.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
</dict>
</plist>
PLIST

echo "==> Ad-hoc code signing"
codesign --force --deep --sign - "$APP"

echo "==> Done: $APP"
