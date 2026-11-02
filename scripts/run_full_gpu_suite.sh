#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# ==============================================================================
# NeuralOps Universal Autonomous Accelerator & Benchmark Suite
#
# Lean, robust, hardware-agnostic (CUDA, MPS, CPU).
#
# Usage:
#   bash scripts/run_full_gpu_suite.sh                  # Default full sweep
#   bash scripts/run_full_gpu_suite.sh --quick          # Fast smoke sweep
#   bash scripts/run_full_gpu_suite.sh --device cuda    # Force CUDA
#   bash scripts/run_full_gpu_suite.sh --steps 100      # Custom training steps
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve Python environment
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(which python3)"
fi

# Default configuration options
DEVICE="auto"
LIMIT=""
STEPS=20
BATCH_SIZE="auto"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            LIMIT="--limit 5"
            STEPS=10
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_full_gpu_suite.sh [--quick] [--device <cuda|mps|cpu|auto>] [--limit <N>] [--steps <N>]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "  🚀 NeuralOps Universal Autonomous Execution Suite"
echo "======================================================================"
echo "  Repository  : ${REPO_ROOT}"
echo "  Interpreter : ${PYTHON_BIN}"
echo "  Target      : ${DEVICE}"
echo "  Timestamp   : $(date)"
echo "======================================================================"

# Ensure results directory exists
mkdir -p "${REPO_ROOT}/results"

# Phase 1: Hardware Pre-Flight & Memory Auto-Tuner
echo ""
echo ">>> [1/4] Hardware Pre-Flight & VRAM Auto-Tuning..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/gpu_telemetry.py" --probe-model

# Phase 2: Master Benchmark Sweep (Attentions, Compressors & Kernel Fallbacks)
echo ""
echo ">>> [2/4] Executing Benchmark Suite (Attentions, Compressors & Kernels)..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/gpu_master_suite.py" --device "${DEVICE}" ${LIMIT}

# Phase 3: Smart GPU-Maximizing Trainer Engine
echo ""
echo ">>> [3/4] Launching Smart GPU-Maximizing Trainer Engine..."
TRAIN_ARGS=(--steps "${STEPS}")
if [ "${BATCH_SIZE}" != "auto" ]; then
    TRAIN_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/smart_gpu_trainer.py" "${TRAIN_ARGS[@]}"

# Phase 4: Structured Telemetry & Auto-Summary Persistence
echo ""
echo ">>> [4/4] Writing Telemetry Ledger & Executive Summaries..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/results_exporter.py" --demo

echo ""
echo "======================================================================"
echo "  ✓ Execution Complete! All artifacts saved to results/:"
echo "    • SQLite DB   : results/master_ledger.sqlite"
echo "    • JSONL Feed  : results/benchmarks_latest.jsonl"
echo "    • CSV Sheet   : results/benchmarks_latest.csv"
echo "    • Summary Doc : results/BENCHMARK_EXECUTIVE_SUMMARY.md"
echo "======================================================================"
