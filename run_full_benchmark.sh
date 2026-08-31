#!/usr/bin/env bash
# ===============================================================================
# NeuralOps: Full End-to-End Two-Phase Attention & Compression Benchmark Suite
# Phase 1: High-Throughput Adaptive Memory-Aware Efficiency Sweep (720 combinations)
# Phase 2: Multi-Seed Statistical Quality Sweep (20 Architectures x 4 Models x 4 GLUE Tasks)
# ===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/Codes:${PYTHONPATH:-}"

echo "==============================================================================="
echo " NeuralOps End-to-End Benchmark Suite (NeurIPS / ICLR Submission Track)"
echo "==============================================================================="

# 1. Phase 1: Efficiency Sweep
echo "[*] Phase 1: Launching Adaptive Efficiency Sweep..."
python -u "${SCRIPT_DIR}/Codes/framework/two_phase_benchmark.py" \
    --mode efficiency \
    --precision fp16 \
    --batch_size 128 \
    --seq_len 128

# 2. Phase 2: Quality & Multi-Seed Statistical Sweep
echo "[*] Phase 2: Launching Multi-Seed Quality Benchmark Suite..."
python -u "${SCRIPT_DIR}/Codes/framework/run_quality_sweep.py" \
    --models "bert,roberta,distilbert,t5" \
    --datasets "sst2,mrpc,cola,stsb" \
    --seeds "42,123,777" \
    --epochs 3 \
    --max_steps 20 \
    --lr 3e-5 \
    --weight_decay 0.01 \
    --precision fp16

echo "[*] Two-Phase Benchmark Completed Successfully!"
