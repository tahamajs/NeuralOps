#!/bin/bash
# Runner for Zero-Shot Reasoning (Category 3)
# Usage: ./runner_benchmark_zeroshot.sh [MODULE_NAME]
MODULE=${1:-"performer_attention"}

echo "Running Zero-Shot LM Harness for $MODULE..."
python3 scripts/run_zeroshot_parallel.py \
  --parallel 4 \
  --num_samples 500 \
  --batch_size 8 \
  --model models/gpt2 \
  --module "$MODULE"
