#!/bin/bash
# Runner for RULER / Needle-In-A-Haystack (Category 6)
# Usage: ./runner_benchmark_ruler.sh [MODULE_NAME]
MODULE=${1:-"simple_ssm"}

echo "Running RULER Benchmark for $MODULE..."
python3 -m neuralops.benchmarks.run_ruler_benchmark \
  --modules "$MODULE" \
  --model models/qwen2.5 \
  --context_size 4096
