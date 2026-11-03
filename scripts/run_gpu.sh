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
MODE="${1:-help}"
shift || true
case "${MODE}" in
    validate-cuda) exec "${PYTHON_BIN}" "${NEURALOPS_ROOT}/scripts/benchmarks/validate_4_cuda_kernels.py" "$@" ;;
    benchmark) exec "${PYTHON_BIN}" "${NEURALOPS_ROOT}/scripts/benchmarks/gpu_master_suite.py" --device cuda:0 "$@" ;;
    triton) exec "${PYTHON_BIN}" "${NEURALOPS_ROOT}/neuralops/benchmarks/benchmark_all_26_comprehensive.py" --device cuda:0 "$@" ;;
    engine) exec "${SCRIPT_DIR}/run_engine_benchmark.sh" "$@" ;;
    full) exec "${SCRIPT_DIR}/run_full_gpu_suite.sh" "$@" ;;
    help|-h|--help)
        cat <<'EOF'
Usage: scripts/run_gpu.sh <validate-cuda|benchmark|triton|engine|full> [options]
EOF
        ;;
    *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
