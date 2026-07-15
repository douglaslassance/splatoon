#!/usr/bin/env bash
#
# Installs the external tools Splatoon's multi-image (scene) reconstruction shells
# out to. Single-image splats (SHARP) need none of this.
#
#   - COLMAP    - camera pose solve (sparse reconstruction)  [Homebrew]
#   - OpenSplat - Gaussian-splat trainer, Apple Metal (MPS)  [built from source]
#   - Brush     - optional native-Metal splat trainer         [cargo]
#   - OpenMVS   - dense MVS for the photogrammetry textured mesh [built from source]
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

# --- OpenMVS (optional, photogrammetry textured mesh) ------------------------
# The "Photogrammetry" mesh method runs COLMAP + OpenMVS on a scene's images to
# build a watertight, UV-textured mesh. Built from source (no Homebrew formula),
# CPU-only (no CUDA on Apple Silicon). Pinned to v2.3.0, which needs only the
# classic deps (Eigen/OpenCV/CGAL/Boost/VCGLib) — later versions pull in a heavy
# vcpkg-only SfM stack we don't use (COLMAP already solves poses).
#
# This recipe was hardened against a very new Homebrew/toolchain (Eigen 5,
# AppleClang 21, Boost 1.90). Each workaround is load-bearing there and harmless
# on older setups:
#   - boost@1.85: Boost 1.90 made boost_system header-only, which v2.3.0 rejects.
#   - vendored Eigen 3.4: Homebrew's eigen may be 5.x, which v2.3.0 refuses; Eigen
#     is header-only, so clone 3.4 and point OpenMVS at it. Its bundled FindEigen3
#     mis-parses non-3.x versions, so we force the Eigen block on (if(TRUE)).
#   - patch std::shared_ptr::unique() (gone in newer libc++) -> use_count() != 1.
#   - -Wno-* : AppleClang 21 promotes some VCGLib warnings to errors.
#   - patch Utils.cmake so libomp is actually linked into the executables.
#   - CMAKE_CXX_STANDARD=17, viewer/python/breakpad off, explicit OpenMP flags.
echo "==> Building OpenMVS (photogrammetry mesh; optional)"
for pkg in cmake opencv cgal glew nanoflann libomp boost@1.85; do
  brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg"
done
B185="$(brew --prefix boost@1.85)"
LIBOMP="$(brew --prefix libomp)"
OMVS_WORK="${TMPDIR:-/tmp}/splatoon-openmvs"
rm -rf "$OMVS_WORK"; mkdir -p "$OMVS_WORK"; cd "$OMVS_WORK"
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git eigen34
git clone --depth 1 https://github.com/cdcseacave/VCG.git vcglib
git clone --depth 1 --branch v2.3.0 https://github.com/cdcseacave/openMVS.git openMVS
# Source patches (see notes above).
sed -i '' 's/if(EIGEN3_FOUND)/if(TRUE)/' openMVS/CMakeLists.txt
sed -i '' 's/!store_\.unique()/store_.use_count() != 1/' openMVS/libs/Common/FastDelegateCPP11.h
sed -i '' "s|set(CMAKE_EXE_LINKER_FLAGS \"-stdlib=libc++\")|set(CMAKE_EXE_LINKER_FLAGS \"-stdlib=libc++ -L$LIBOMP/lib -lomp -Wl,-rpath,$LIBOMP/lib\")|" openMVS/build/Utils.cmake
mkdir -p openMVS_build && cd openMVS_build
cmake ../openMVS -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_FLAGS="-Wno-error -Wno-missing-template-arg-list-after-template-kw" \
  -DOpenMVS_USE_CUDA=OFF -DOpenMVS_USE_PYTHON=OFF \
  -DOpenMVS_USE_BREAKPAD=OFF -DOpenMVS_BUILD_VIEWER=OFF \
  -DVCG_ROOT="$OMVS_WORK/vcglib" \
  -DBOOST_ROOT="$B185" -DBoost_DIR="$B185/lib/cmake/Boost-1.85.0" \
  -DEIGEN3_INCLUDE_DIR="$OMVS_WORK/eigen34" -DEIGEN3_INCLUDE_DIRS="$OMVS_WORK/eigen34" \
  -DOpenMP_C_FLAGS="-Xclang -fopenmp -I$LIBOMP/include" \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I$LIBOMP/include" \
  -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$LIBOMP/lib/libomp.dylib" \
  -DCMAKE_PREFIX_PATH="$(brew --prefix opencv);$(brew --prefix cgal);$(brew --prefix nanoflann);$B185"
cmake --build . --config Release -j"$(sysctl -n hw.ncpu)"
# Install into ~/.local/bin (the standard user-tools dir, on Splatoon's search
# path) rather than Homebrew's managed bin — these aren't brew-managed, so they
# don't belong there, and this keeps them trivially removable.
USER_BIN="$HOME/.local/bin"; mkdir -p "$USER_BIN"
for bin in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh; do
  found="$(find "$OMVS_WORK/openMVS_build" -maxdepth 3 -type f -name "$bin" | head -1)"
  if [ -n "$found" ]; then echo "==> Installing $bin -> $USER_BIN/$bin"; cp "$found" "$USER_BIN/$bin"; fi
done

# --- TripoSplat CLI (optional, single-image 3D object generator) -------------
# The "TripoSplat" single-image generator turns one photo into a complete 3D
# object (vs SHARP's 2.5D relief). It's a Python/PyTorch tool installed with uv
# as an isolated CLI on PATH (~/.local/bin, on Splatoon's search path). Weights
# (~3.8 GB) are fetched separately with `tripo-cli download`.
echo "==> Installing TripoSplat CLI (optional, single-image 3D objects)"
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"; brew list uv >/dev/null 2>&1 || brew install uv
fi
TRIPO_CLI_DIR="$(cd "$(dirname "$0")/../../tripo-cli" 2>/dev/null && pwd || true)"
if [ -n "$TRIPO_CLI_DIR" ] && [ -f "$TRIPO_CLI_DIR/pyproject.toml" ]; then
  uv tool install --force "$TRIPO_CLI_DIR"
  echo "==> Fetching TripoSplat weights (~3.8 GB)"
  tripo-cli download
else
  echo "==> Skipping TripoSplat: ../tripo-cli not found next to the Splatoon repo." >&2
  echo "    Clone it, then run: uv tool install <path-to-tripo-cli> && tripo-cli download" >&2
fi

echo ""
echo "Done."
echo "  COLMAP:    $(command -v colmap)"
echo "  OpenSplat: $DEST/opensplat"
echo "  OpenMVS:   $HOME/.local/bin/DensifyPointCloud (+ ReconstructMesh, TextureMesh, …)"
echo "  TripoSplat: $(command -v tripo-cli || echo 'not installed')"
echo "Open a photo that has several same-place/time siblings to reconstruct a scene."
