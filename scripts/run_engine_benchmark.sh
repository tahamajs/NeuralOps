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
ENGINE="pytorch"
MODEL="gpt2"
SEQ_LEN="128"
OUTPUT="${NEURALOPS_ROOT}/results/serving_benchmark.json"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine) ENGINE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        -h|--help) echo "Usage: scripts/run_engine_benchmark.sh [--engine all|pytorch|vllm|sglang|tensorrt_llm] [--model MODEL] [--seq-len N] [--output PATH]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$(dirname "${OUTPUT}")"
exec "${PYTHON_BIN}" "${NEURALOPS_ROOT}/scripts/benchmark_serving.py" --engine "${ENGINE}" --model "${MODEL}" --seq-len "${SEQ_LEN}" --output "${OUTPUT}"
