#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/lib/neuralops_env.sh"
MODE="${1:-help}"
shift || true
case "${MODE}" in
    kernels) exec "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/benchmark_all_26_comprehensive.py" "$@" ;;
    cuda) exec "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/benchmark_all_108_cuda_isolated.py" "$@" ;;
    attention) exec "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/benchmark_torch_compile_131_attentions.py" "$@" ;;
    systems) exec "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/benchmark_engine_matrix.py" "$@" ;;
    numerics) exec "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/measure_multiprecision.py" "$@" ;;
    resources) exec "${PYTHON_BIN}" -m neuralops.benchmarks.resource_scaling "$@" ;;
    dispatch) exec "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/benchmark_dispatch_integrity.py" "$@" ;;
    all)
        "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/benchmark_dispatch_integrity.py" "$@"
        "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/benchmark_all_26_comprehensive.py" "$@"
        exec "${PYTHON_BIN}" "${ROOT_DIR}/benchmarks/benchmark_torch_compile_131_attentions.py" "$@"
        ;;
    help|-h|--help) echo "Usage: benchmarks/run_suite.sh <kernels|cuda|attention|systems|numerics|resources|dispatch|all> [options]" ;;
    *) echo "Unknown benchmark group: ${MODE}" >&2; exit 2 ;;
esac
