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

echo "NeuralOps root: ${NEURALOPS_ROOT}"
echo "Python: ${PYTHON_BIN}"
command -v ruff || echo "ruff: not found"
command -v qsub || echo "qsub: not found (local-only environment)"
command -v screen || echo "screen: not found (required for detached HPC submission)"
command -v nvidia-smi || echo "nvidia-smi: not found (GPU checks require a CUDA node)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi
if [[ -d "${NEURALOPS_ROOT}/.git" ]]; then
    git -C "${NEURALOPS_ROOT}" diff --stat
fi
