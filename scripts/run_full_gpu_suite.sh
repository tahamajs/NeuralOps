#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# ==============================================================================
# NeuralOps Master Autonomous GPU Execution Suite
#
# Executes the full 5-phase suite:
# 1. Hardware Pre-Flight & VRAM Auto-Tuning
# 2. Unified GPU Master Benchmark (Attentions, Compressors, Fallbacks)
# 3. Smart GPU-Maximizing Trainer Engine
# 4. Atomic Results Persistence to results/master_ledger.sqlite & summaries
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "  🚀 NeuralOps Master Autonomous Accelerator Execution Suite"
echo "======================================================================"
echo "  Repository Root : ${REPO_ROOT}"
echo "  Python Binary   : ${PYTHON_BIN}"
echo "  Date/Time       : $(date)"
echo "======================================================================"

# Phase 1: Hardware Pre-Flight & Telemetry
echo ""
echo ">>> [Phase 1/4] Running Hardware Pre-Flight & VRAM Auto-Tuner..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/gpu_telemetry.py" --probe-model

# Phase 2: Master Benchmark Suite
echo ""
echo ">>> [Phase 2/4] Executing Master Benchmark Suite (Attentions & Compressors)..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/gpu_master_suite.py" --limit 10

# Phase 3: Smart GPU-Maximizing Trainer
echo ""
echo ">>> [Phase 3/4] Launching Smart GPU-Maximizing Trainer Engine..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/smart_gpu_trainer.py" --steps 20 --batch-size 8

# Phase 4: Atomic Results Persistence
echo ""
echo ">>> [Phase 4/4] Finalizing Results Ledger & Auto-Summaries..."
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/results_exporter.py" --demo

echo ""
echo "======================================================================"
echo "  🎉 Master GPU Suite Successfully Completed!"
echo "  Results stored in: ${REPO_ROOT}/results/"
echo "    • SQLite Ledger  : results/master_ledger.sqlite"
echo "    • JSONL Telemetry: results/benchmarks_latest.jsonl"
echo "    • CSV Summary    : results/benchmarks_latest.csv"
echo "    • Executive Report: results/BENCHMARK_EXECUTIVE_SUMMARY.md"
echo "======================================================================"
