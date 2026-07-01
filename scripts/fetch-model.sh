#!/usr/bin/env bash
#
# Downloads the SHARP Core ML model (~2.7 GB) from Hugging Face into ./Models.
# The model is not committed to git. After downloading, launch Splatoon and,
# when prompted, point it at Models/sharp.mlpackage (remembered thereafter).
#
# Requires the Hugging Face CLI:
#   pip install huggingface-hub
# The repo uses Xet storage; huggingface-cli handles it transparently.

set -euo pipefail

REPO="pearsonkyle/Sharp-coreml"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/Models"

if command -v hf >/dev/null 2>&1; then
  HF="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF="huggingface-cli"
else
  echo "error: Hugging Face CLI not found. Install it with:" >&2
  echo "         pip install -U huggingface-hub" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

echo "Downloading sharp.mlpackage from $REPO into $DEST_DIR ..."
"$HF" download "$REPO" --include "sharp.mlpackage/*" --local-dir "$DEST_DIR"

echo ""
echo "Done. Model at: $DEST_DIR/sharp.mlpackage"
echo "Launch Splatoon, click Generate Splat, and select that file when prompted."
