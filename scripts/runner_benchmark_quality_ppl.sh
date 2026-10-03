#!/bin/bash
# Runner for Quality / Perplexity Sweeps (Category 4)
# Usage: ./runner_benchmark_quality_ppl.sh [MODULE_NAME]
MODULE=${1:-"asvd_attention"}

echo "Running Quality & Perplexity (C4) Sweep for $MODULE..."
python3 -m neuralops.benchmarks.run_quality_sweep \
  --modules "$MODULE" \
  --model models/gpt2 \
  --model_family gpt2 \
  --dataset c4 \
  --rank 256 \
  --batch_size 8
