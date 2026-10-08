#!/bin/bash
# Full 232-module benchmark. Run on GPU server only.
set -e

OUT_DIR="results/full_232_$(date +%Y%m%d_%H%M%S)"
echo "Output: $OUT_DIR"

python -m neuralops.benchmarks run \
    --all \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 32 \
    --epochs 1 \
    --batch-size 32 \
    --seq-len 128 \
    --parallel 1 \
    --output-dir "$OUT_DIR" \
    --resume

echo "Done. Results in: $OUT_DIR"
python -m neuralops.benchmarks results --output-dir "$OUT_DIR" --summary
