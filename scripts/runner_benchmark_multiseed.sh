#!/bin/bash
# Runner for Multi-Seed Robustness Finetuning (Category 5)
# Usage: ./runner_benchmark_multiseed.sh [MODULE_NAME]
MODULE=${1:-"svd_linear_attention"}

echo "Running Multi-Seed Evaluation for $MODULE..."
python3 -m neuralops.benchmarks.run_multiseed \
  --modules "$MODULE" \
  --models distilbert \
  --dataset sst2 \
  --seeds 42,123,456,789,999
