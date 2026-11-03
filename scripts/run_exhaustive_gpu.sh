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
PYTEST_ARGS=(tests/gpu_exhaustive -v --tb=short)
if [[ $# -gt 0 ]]; then
    PYTEST_ARGS+=("$@")
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" && "${NEURALOPS_ALLOW_CPU_EXHAUSTIVE:-0}" != "1" ]]; then
    echo "Refusing exhaustive suite without CUDA_VISIBLE_DEVICES; submit to a GPU node." >&2
    exit 2
fi
exec "${PYTHON_BIN}" -m pytest "${PYTEST_ARGS[@]}"
