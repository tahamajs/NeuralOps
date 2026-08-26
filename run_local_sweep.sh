#!/usr/bin/env bash
# ==============================================================================
# Local Benchmark Runner for NeuralOps Framework (Mac / Apple Silicon / CPU)
# Runs the benchmark suite locally without requiring SSH to the GPU server.
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}/Codes:${PYTHONPATH}"

echo "======================================================================"
echo "⚡ Starting NeuralOps Benchmark Suite Locally"
echo "   Directory: ${PROJECT_DIR}"
echo "   Platform:  $(uname -sm)"
echo "======================================================================"

python3 -u Codes/framework/launch_gpu_server.py \
    --mode fast \
    --precision fp32 \
    --models bert,roberta \
    --datasets sst2 \
    --batch_size 16 \
    --seq_len 64 \
    --max_steps 3 \
    --output_dir results/local \
    "$@" 2>&1 | tee "${PROJECT_DIR}/benchmark_local.log"
