# tripo-cli

A small command-line wrapper around [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)
(TripoAI / VAST-AI) that turns a **single image into a 3D Gaussian splat** (`.ply`),
runnable on Apple Silicon (MPS). Built to be shelled out to from a host app.

Managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync                    # create the venv and install torch + deps
uv run tripo-cli download  # fetch the ~3.8 GB weights from Hugging Face
```

Weights land in `~/.cache/triposplat/ckpts` by default (override with
`--ckpt-dir` or the `TRIPOSPLAT_CKPTS` env var).

## Usage

```bash
uv run tripo-cli generate input.jpg output.ply \
    --device mps --num-gaussians 131072
```

- `--device auto|mps|cuda|cpu` (default `auto`)
- `--num-gaussians N` (default 131072; up to 262144)
- `--steps N` diffusion steps (default 20)
- `--splat` also write a `.splat`
- Progress is printed to **stderr** as `Sampling: N%` for a caller to parse.

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so the one MPS-unsupported
op (`deform_conv2d`) falls back to CPU instead of crashing.

## Vendored code & license

`src/tripo_cli/triposplat.py` and `src/tripo_cli/model.py` are vendored verbatim
from TripoSplat (only the internal `from model import` was made relative). TripoSplat
is MIT-licensed; see its repository for the original source and weights license.
This wrapper is provided under the same terms.
