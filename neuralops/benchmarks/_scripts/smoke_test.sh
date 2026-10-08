#!/bin/bash
# Smoke test for benchmark interface — run on GPU server only.
set -e

echo "=== Test 1: List all modules ==="
python -m neuralops.benchmarks list --type attention | head -20

echo "=== Test 2: Plan a single run ==="
python -m neuralops.benchmarks plan \
    --module svd_linear_attention \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64

echo "=== Test 3: Baseline (no compression, 1 epoch) ==="
python -m neuralops.benchmarks run \
    --no-compression \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --epochs 1 \
    --batch-size 32 \
    --seq-len 128 \
    --output-dir results/smoke_baseline

echo "=== Test 4: One attention module (3 epochs) ==="
python -m neuralops.benchmarks run \
    --module svd_linear_attention \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64 \
    --epochs 3 \
    --batch-size 32 \
    --seq-len 128 \
    --output-dir results/smoke_svd

echo "=== Test 5: One compressor (3 epochs) ==="
python -m neuralops.benchmarks run \
    --module direct_svd \
    --model distilbert-base-uncased \
    --dataset sst2 \
    --rank 64 \
    --epochs 3 \
    --batch-size 32 \
    --seq-len 128 \
    --output-dir results/smoke_direct_svd

echo "=== Verify results ==="
python -m neuralops.benchmarks results --output-dir results/smoke_baseline --summary
python -m neuralops.benchmarks results --output-dir results/smoke_svd --summary
python -m neuralops.benchmarks results --output-dir results/smoke_direct_svd --summary

echo "=== Expected: baseline > 0.80, svd > 0.75, direct_svd > 0.75 ==="
