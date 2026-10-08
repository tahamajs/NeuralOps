#!/bin/bash
# Benchmark the 14 missing modules on GPU.

set -e

OUT_DIR="results/08_missing_modules_$(date +%Y%m%d)"

# Modules missing
MODULES=(
    "grouped_query_attention"
    "adaptive_sparse_attn"
    "x_attention"
    "svd_linear_attn"
    "simplessm"
    "selective_mamba"
    "hgrn2_gated_outer"
    "hymba_transx_attn"
    "fused_svd_kernel_attention"
    "inject_triton_cosformer"
    "inject_triton_svd_linear_attention"
    "triton_cosformer_inject"
    "triton_svd_linear_attention_inject"
    "inject_xattention"
)

for module in "${MODULES[@]}"; do
    echo "=== Benchmarking: $module ==="
    python -m neuralops.benchmarks run \
        --module "$module" \
        --model distilbert-base-uncased \
        --dataset sst2 \
        --rank 64 \
        --epochs 3 \
        --output-dir "$OUT_DIR/$module" \
        || echo "FAILED: $module"
done

echo "Done. Results in $OUT_DIR"
