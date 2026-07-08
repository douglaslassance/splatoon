#!/usr/bin/env bash
#
# Installs the external tools Splatoon's multi-image (scene) reconstruction shells
# out to. Single-image splats (SHARP) need none of this.
#
#   - COLMAP    - camera pose solve (sparse reconstruction)  [Homebrew]
#   - OpenSplat - Gaussian-splat trainer, Apple Metal (MPS)  [built from source]
#
# COLMAP installs into Homebrew's bin; OpenSplat's binary is copied to
# /usr/local/bin. Both directories are on Splatoon's tool search path (see
# ToolLocator.swift). If you install either elsewhere, point Splatoon at it with
# the SPLATOON_COLMAP / SPLATOON_OPENSPLAT environment variables, or via the
# open panel Splatoon shows the first time it needs a tool.
#
# NOTE: the OpenSplat build links libtorch and can take a while. The exact cmake
# flags below (libtorch prefix, GPU runtime) may need adjusting for your setup;
# see https://github.com/pierotofy/OpenSplat for authoritative build steps.

set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "error: Homebrew is required (https://brew.sh)." >&2
  exit 1
fi

# --- COLMAP (reliable) -------------------------------------------------------
echo "==> Installing COLMAP"
brew list colmap >/dev/null 2>&1 || brew install colmap

# --- OpenSplat build dependencies -------------------------------------------
echo "==> Installing OpenSplat build dependencies (cmake, opencv, pytorch)"
for pkg in cmake opencv pytorch; do
  brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg"
done

# libtorch's CMake package ships inside Homebrew's pytorch, several levels deep
# under site-packages (not at `brew --prefix pytorch`), so locate TorchConfig.cmake
# and hand cmake the torch root that contains it.
PYT="$(realpath "$(brew --prefix pytorch)")"
TORCH_CONFIG="$(find "$PYT" -name TorchConfig.cmake 2>/dev/null | head -1)"
if [ -z "$TORCH_CONFIG" ]; then
  echo "error: TorchConfig.cmake not found under $PYT" >&2
  exit 1
fi
TORCH_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$TORCH_CONFIG")")")")"
OPENCV_PREFIX="$(brew --prefix opencv)"

# --- Build OpenSplat ---------------------------------------------------------
WORK="${TMPDIR:-/tmp}/splatoon-opensplat"
rm -rf "$WORK"
mkdir -p "$WORK"
echo "==> Cloning OpenSplat into $WORK"
git clone --depth 1 https://github.com/pierotofy/OpenSplat "$WORK/OpenSplat"

cd "$WORK/OpenSplat"
mkdir -p build && cd build
echo "==> Configuring (Metal/MPS build)"
cmake .. \
  -DGPU_RUNTIME=MPS \
  -DCMAKE_PREFIX_PATH="$TORCH_ROOT" \
  -DOPENCV_DIR="$OPENCV_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release
echo "==> Building (this can take several minutes)"
cmake --build . --config Release -j"$(sysctl -n hw.ncpu)"

# --- Install -----------------------------------------------------------------
# Homebrew's bin is writable and already on Splatoon's tool search path.
DEST="$(brew --prefix)/bin"
echo "==> Installing opensplat -> $DEST/opensplat"
mkdir -p "$DEST"
cp opensplat "$DEST/opensplat"

echo ""
echo "Done."
echo "  COLMAP:    $(command -v colmap)"
echo "  OpenSplat: $DEST/opensplat"
echo "Open a photo that has several same-place/time siblings to reconstruct a scene."
