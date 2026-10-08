#!/bin/bash
# GPU smoke test for benchmark interface.
# Run this on the GPU server only.
set -e

OUT_BASE="${1:-results/smoke_$(date +%Y%m%d_%H%M%S)}"
echo "Output directory: $OUT_BASE"

echo ""
echo "═══════════════════════════════════════════════"
echo "  STEP 1: Environment check"
echo "═══════════════════════════════════════════════"
python -m neuralops.benchmarks verify

echo ""
echo "═══════════════════════════════════════════════"
echo "  STEP 2: Baseline (no compression, 1 epoch)"
echo "═══════════════════════════════════════════════"
python -m neuralops.benchmarks run \
    --no-compression \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --epochs 1 \
    --batch-size 32 \
    --seq-len 128 \
    --output-dir "$OUT_BASE/baseline"

echo ""
echo "═══════════════════════════════════════════════"
echo "  STEP 3: One attention module (3 epochs)"
echo "═══════════════════════════════════════════════"
python -m neuralops.benchmarks run \
    --module svd_linear_attention \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64 \
    --epochs 3 \
    --batch-size 32 \
    --seq-len 128 \
    --output-dir "$OUT_BASE/svd"

echo ""
echo "═══════════════════════════════════════════════"
echo "  STEP 4: One compressor (3 epochs)"
echo "═══════════════════════════════════════════════"
python -m neuralops.benchmarks run \
    --module direct_svd \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64 \
    --epochs 3 \
    --batch-size 32 \
    --seq-len 128 \
    --output-dir "$OUT_BASE/compressor"

echo ""
echo "═══════════════════════════════════════════════"
echo "  STEP 5: Results summary"
echo "═══════════════════════════════════════════════"
python -m neuralops.benchmarks results --output-dir "$OUT_BASE/baseline" --summary
python -m neuralops.benchmarks results --output-dir "$OUT_BASE/svd" --summary
python -m neuralops.benchmarks results --output-dir "$OUT_BASE/compressor" --summary

echo ""
echo "═══════════════════════════════════════════════"
echo "  EXPECTED: baseline > 0.80, svd > 0.75, direct_svd > 0.75"
echo "═══════════════════════════════════════════════"
