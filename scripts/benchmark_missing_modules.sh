#!/bin/bash
# Benchmark suite runner using the unified CLI
set -e

MODULES=(
    "gqa_attention"
    "adaptive_sparse_attention"
    "xattention"
    "svd_linear_attention"
    "simple_ssm"
    "hgrn2_gated_outer"
    "hymba_transx"
)

for module in "${MODULES[@]}"; do
    echo "=== Benchmarking: $module ==="
    python3 -m neuralops bench --module "$module" --mode eval "$@" || echo "FAILED: $module"
done
