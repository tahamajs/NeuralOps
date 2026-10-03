#!/bin/bash
# Runner for Rank Sweeps (Category 2)
# Usage: ./runner_benchmark_rank_sweep.sh [MODULE_NAME] [DATASET]
MODULE=${1:-"dobi_svd"}
DATASET=${2:-"mnli"}

echo "Running Rank Sweep for $MODULE on $DATASET..."
python3 -m neuralops.benchmarks.run_rank_sweeper \
  --modules "$MODULE" \
  --ranks 64,128,256,384 \
  --model "models/base_checkpoints/distilbert_$DATASET" \
  --model_family bert \
  --dataset "$DATASET" \
  --batch_size 16
