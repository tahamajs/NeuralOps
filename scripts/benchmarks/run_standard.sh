#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/lib/neuralops_env.sh"
MODE="${1:-help}"
shift || true
case "${MODE}" in
    validate)
        exec "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_4_cuda_kernels.py" "$@"
        ;;
    attention)
        exec "${PYTHON_BIN}" -m neuralops.benchmarks.cli run --category attention --device cuda "$@"
        ;;
    compressor)
        exec "${PYTHON_BIN}" -m neuralops.benchmarks.cli run --category compressor --device cuda "$@"
        ;;
    cuda)
        exec "${PYTHON_BIN}" -m neuralops.benchmarks.cli run --category cuda --device cuda "$@"
        ;;
    triton)
        exec "${PYTHON_BIN}" -m neuralops.benchmarks.cli run --category triton --device cuda "$@"
        ;;
    all)
        exec "${PYTHON_BIN}" -m neuralops.benchmarks.cli run --category all --device cuda "$@"
        ;;
    help|-h|--help)
        cat <<'EOF'
Usage: scripts/benchmarks/run_standard.sh <validate|attention|compressor|cuda|triton|all> [options]

This is the standard benchmark entry point. Use legacy scripts only when
reproducing an historical experiment.
EOF
        ;;
    *) echo "Unknown benchmark group: ${MODE}" >&2; exit 2 ;;
esac
