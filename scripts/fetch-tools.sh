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

# NOTE: The optional "Global pose solver" setting uses COLMAP's own global SfM
# (`colmap global_mapper`, the upstreamed successor to GLOMAP), so it needs no
# extra tool — any COLMAP recent enough to ship that subcommand supports it.

# --- Brush (optional, native-Metal trainer) ----------------------------------
# Selected by the "Trainer" setting. Native wgpu/Metal (no libtorch, no CPU
# fallback), usually much faster on Apple GPUs. Built with cargo into
# ~/.cargo/bin, which Splatoon's tool search path includes. Its headless binary
# only exists on `main` (post-v0.3.0), and `--locked` is required so it builds
# against the burn version it was tested with.
if command -v cargo >/dev/null 2>&1; then
  echo "==> Building Brush (brush-cli) with cargo"
  cargo install --git https://github.com/ArthurBrussee/brush --locked --bin brush-cli brush-cli
else
  echo "==> Skipping Brush: cargo (Rust) not found. Install Rust from https://rustup.rs to enable the Brush trainer." >&2
fi

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

# Load images serially. OpenSplat loads every camera's image in parallel, and
# each thread creates a torch tensor — but PyTorch's MPS backend is not
# thread-safe for concurrent allocation, so the parallel load SIGSEGVs during
# setup with many images (reproduced with pytorch 2.12 on Apple Silicon; the
# CPU backend is unaffected). Loading is a one-time setup cost, so serializing it
# removes the crash while keeping training on the GPU.
echo "==> Patching OpenSplat to load images serially (MPS thread-safety)"
perl -0777 -pi -e 's/parallel_for\(inputData\.cameras\.begin\(\), inputData\.cameras\.end\(\), \[&downScaleFactor\]\(Camera &cam\)\{\s*cam\.loadImage\(downScaleFactor\);\s*\}\);/for (Camera \&cam : inputData.cameras) { cam.loadImage(downScaleFactor); }/s' opensplat.cpp
grep -q "for (Camera &cam : inputData.cameras)" opensplat.cpp \
  || { echo "error: OpenSplat image-load patch did not apply (upstream code changed?)." >&2; exit 1; }

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
