"""tripo-cli: single image -> 3D Gaussian splat (.ply), via TripoSplat.

Designed to be shelled out to from a host app (e.g. Splatoon). Progress is
printed to stderr as `Sampling: N%` (tqdm) so the caller can parse it.
"""
import argparse
import os
import sys
from pathlib import Path

# Unsupported MPS ops (e.g. torchvision deform_conv2d) fall back to CPU instead
# of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# The five weight files, relative to the checkpoint directory.
WEIGHTS = {
    "ckpt_path":              "diffusion_models/triposplat_fp16.safetensors",
    "decoder_path":           "vae/triposplat_vae_decoder_fp16.safetensors",
    "dinov3_path":            "clip_vision/dino_v3_vit_h.safetensors",
    "flux2_vae_encoder_path": "vae/flux2-vae.safetensors",
    "rmbg_path":              "background_removal/birefnet.safetensors",
}
HF_REPO = "VAST-AI/TripoSplat"


def default_ckpt_dir() -> Path:
    env = os.environ.get("TRIPOSPLAT_CKPTS")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "triposplat" / "ckpts"


def resolve_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cmd_download(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download
    dest = Path(args.ckpt_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_REPO} -> {dest}", file=sys.stderr)
    snapshot_download(repo_id=HF_REPO, local_dir=str(dest))
    missing = [p for p in WEIGHTS.values() if not (dest / p).exists()]
    if missing:
        print(f"error: missing after download: {missing}", file=sys.stderr)
        return 1
    print("Download complete.", file=sys.stderr)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    ckpt_dir = Path(args.ckpt_dir).expanduser()
    paths = {k: ckpt_dir / v for k, v in WEIGHTS.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("error: missing weights (run `tripo-cli download` first):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    print(f"device={device} num_gaussians={args.num_gaussians} steps={args.steps}", file=sys.stderr)

    from .triposplat import TripoSplatPipeline
    pipe = TripoSplatPipeline(**{k: str(p) for k, p in paths.items()}, device=device)
    gaussian, prepared = pipe.run(
        args.image,
        seed=args.seed,
        steps=args.steps,
        num_gaussians=args.num_gaussians,
        show_progress=True,
    )

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    gaussian.save_ply(str(out))
    if args.splat:
        gaussian.save_splat(str(out.with_suffix(".splat")))
    if args.save_preprocessed:
        prepared.save(str(out.with_name(out.stem + "_input.webp")))
    print(f"OK -> {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tripo-cli", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="image -> 3D Gaussian splat (.ply)")
    g.add_argument("image", help="input image path")
    g.add_argument("out", help="output .ply path")
    g.add_argument("--ckpt-dir", default=str(default_ckpt_dir()))
    g.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    g.add_argument("--num-gaussians", type=int, default=131072)
    g.add_argument("--steps", type=int, default=20)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--splat", action="store_true", help="also write a .splat file")
    g.add_argument("--save-preprocessed", action="store_true",
                   help="also write the background-removed input next to the output")
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("download", help="download TripoSplat weights from Hugging Face")
    d.add_argument("--ckpt-dir", default=str(default_ckpt_dir()))
    d.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
