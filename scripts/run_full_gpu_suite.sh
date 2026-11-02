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
# 7-Phase Architecture:
# Phase 1: Hardware Pre-Flight & VRAM Auto-Tuning
# Phase 2: Master Attention & Compressor Sweeps (p50/p95/p99 + Parallel)
# Phase 2.5: Triton & CUDA 3-Level Fallback Kernel Leaderboard
# Phase 3: Smart GPU-Maximizing Trainer Engine (WSD + Checkpointing)
# Phase 4: Serving Frameworks & Serialization Verifier (9 Backends)
# Phase 5: Structured Results Persistence (SQLite WAL, Parquet, LaTeX)
# Phase 6: Publication-Grade Visualizer (Roofline, Pareto)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(which python3)"
fi

DEVICE="auto"
LIMIT=""
STEPS=20
BATCH_SIZE="auto"
BACKWARD_FLAG=""

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
        --backward)
            BACKWARD_FLAG="--backward"
            shift
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
            echo "Usage: bash scripts/run_full_gpu_suite.sh [--quick] [--device <cuda|mps|cpu|auto>] [--limit <N>] [--backward] [--steps <N>] [--batch-size <N>]"
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

mkdir -p "${REPO_ROOT}/results/tables"
mkdir -p "${REPO_ROOT}/results/figures"

# Phase 1: Hardware Pre-Flight & VRAM Auto-Tuning
echo ""
echo ">>> [1/6] Hardware Pre-Flight & VRAM Auto-Tuning..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/gpu_telemetry.py" --json

# Phase 2: Master Benchmark Sweep (Attentions & Compressors)
echo ""
echo ">>> [2/6] Executing High-Precision Benchmark Suite..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/gpu_master_suite.py" --device "${DEVICE}" ${LIMIT} ${BACKWARD_FLAG} --export

# Phase 2.5: Triton / CUDA Kernel Leaderboard
echo ""
echo ">>> [2.5/6] Probing 3-Level Fallback Kernel Leaderboard..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/triton_leaderboard.py" --device "${DEVICE}" --limit 5

# Phase 3: Smart GPU-Maximizing Trainer Engine
echo ""
echo ">>> [3/6] Launching Smart GPU-Maximizing Trainer Engine..."
TRAIN_ARGS=(--steps "${STEPS}")
if [ "${BATCH_SIZE}" != "auto" ]; then
    TRAIN_ARGS+=(--batch-size "${BATCH_SIZE}")
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/smart_gpu_trainer.py" "${TRAIN_ARGS[@]}"

# Phase 4: Serving Frameworks & Serialization Verification (9 Backends)
echo ""
echo ">>> [4/6] Verifying 9 Serving Backends & Serialization Targets..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/serving_matrix_verifier.py" --ci

# Phase 5: Structured Telemetry & Auto-Summary Persistence
echo ""
echo ">>> [5/6] Writing Telemetry Ledger & Executive Summaries..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/results_exporter.py" --from-jsonl "${REPO_ROOT}/results/benchmarks_latest.jsonl"

# Phase 6: Publication-Grade Visualizations
echo ""
echo ">>> [6/6] Generating Publication Visualizations..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/visualizer.py" --demo

echo ""
echo "======================================================================"
echo "  ✓ Execution Complete! All artifacts saved to results/:"
echo "    • SQLite DB   : results/master_ledger.sqlite (WAL mode)"
echo "    • JSONL Feed  : results/benchmarks_latest.jsonl"
echo "    • CSV Sheet   : results/benchmarks_latest.csv"
echo "    • Parquet     : results/benchmarks_latest.parquet"
echo "    • Markdown    : results/BENCHMARK_EXECUTIVE_SUMMARY.md"
echo "    • LaTeX Table : results/tables/benchmark_summary.tex"
echo "    • Roofline SVG: results/figures/roofline.svg"
echo "    • Pareto SVG  : results/figures/pareto_compressors.svg"
echo "======================================================================"
