#!/bin/bash
# Runner for Sequence Length Scaling (Category 6)
# Usage: ./runner_benchmark_seqlen_scaling.sh [MODULE_NAME]
MODULE=${1:-"linformer_attention"}

echo "Running Sequence Length Scaling Benchmark for $MODULE..."
python3 -m neuralops.benchmarks.run_seqlen_scaling \
  --modules "$MODULE" \
  --model models/distilbert_sst2 \
  --dataset sst2
