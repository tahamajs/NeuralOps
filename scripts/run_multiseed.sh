#!/usr/bin/env bash
# run_multiseed.sh – ران کردن Multi‑seed برای ماژول‌های top‑10
# استفاده: ./scripts/run_multiseed.sh

set -euo pipefail

BASE_DIR=$(dirname "$(realpath "$0")")/..   # ریشه پروژه
cd "$BASE_DIR"

# لیست ماژول‌های هدف
MODULES=(
    flash3
    block_transformer_hiba
    sdrm
    huffman
    kmeans
    span_expanded
    mlorq
    hybridkv
    fusion_attn
    full
)

SEEDS=(1 2024)   # به‌جز seed 42 که در run اصلی هست

for SEED in "${SEEDS[@]}"; do
    echo "=== Seed $SEED ==="
    python Codes/framework/main.py sweep \
        --attn_types ${MODULES[@]} \
        --task sst2 \
        --seeds $SEED \
        --model_name distilbert-base-uncased \
        --batch_size 64 \
        --epochs 1 \
        --lr 3e-5 \
        --max_len 128 \
        --output_dir results/multiseed_run \
        --exp_name multiseed_s${SEED}
done

echo "✅ تمام Multi‑seed اجرا شد. نتایج در results/multiseed_run/*/summary.csv"
