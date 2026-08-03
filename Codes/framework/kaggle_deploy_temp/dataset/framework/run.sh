#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Efficient Attention Framework — convenience runner
# Usage: bash run.sh <command> [args...]
#
# Examples:
#   bash run.sh train --attn_type full --task sst2 --epochs 3
#   bash run.sh train --config configs/default.yaml --attn_type nha
#   bash run.sh eval  --attn_type full --task sst2 --model_path results/full_sst2_s42/model
#   bash run.sh compress --compressor svd --task sst2
#   bash run.sh sweep --attn_types full nha lsh --seeds 42 123 --task sst2
#   bash run.sh list
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f "$SCRIPT_DIR/../../venv/bin/activate" ]; then
    source "$SCRIPT_DIR/../../venv/bin/activate"
    echo "[run.sh] venv activated"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "[run.sh] Running: python main.py $*"
python main.py "$@"
