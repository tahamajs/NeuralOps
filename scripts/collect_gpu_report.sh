#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/neuralops_env.sh"
REPORT_DIR="${NEURALOPS_ROOT}/results/reports"
mkdir -p "${REPORT_DIR}"
REPORT="${REPORT_DIR}/gpu_report_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "NeuralOps GPU report"
    date -u
    git -C "${NEURALOPS_ROOT}" rev-parse --short HEAD
    git -C "${NEURALOPS_ROOT}" status --short
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
    else
        echo "nvidia-smi unavailable"
    fi
    find "${NEURALOPS_ROOT}/results" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -50
} | tee "${REPORT}"
echo "Saved report: ${REPORT}"
