#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -euo pipefail
NEURALOPS_ROOT="${NEURALOPS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export NEURALOPS_ROOT PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONPATH="${NEURALOPS_ROOT}:${PYTHONPATH:-}"
if [[ -n "${NEURALOPS_PYTHON:-}" ]]; then
    PYTHON_BIN="${NEURALOPS_PYTHON}"
elif [[ -x "${NEURALOPS_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${NEURALOPS_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found; set NEURALOPS_PYTHON." >&2
    return 1
fi
export PYTHON_BIN
mkdir -p "${NEURALOPS_ROOT}/results/logs"
