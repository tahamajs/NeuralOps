# Results Fix Report

Comprehensive report on data quality remediation, normalization, duplicate archival, and coverage gap analysis.

## Fixed Items

- **Invalid JSON**: Repaired 34 malformed JSON files down to **0**.
  - 24 files with truncated `tokenizer:` keys restored and closed.
  - 9 files with concatenated JSON objects / duplicate closing brackets parsed cleanly with `raw_decode`.
  - 1 file with trailing comma syntax error in deep sweep checkpoint fixed.
- **Schema Normalization**: Normalized **16,098** result JSON files into `_MASTER_SUMMARY.csv` with uniform columns (`source_file`, `module`, `model`, `dataset`, `rank`, `accuracy`, `f1`, `loss`, `perplexity`, `peak_vram_mb`, `latency_ms`, `duration_s`, `compression_ratio`, `status`).
- **Duplicate Combinations**: Isolated **607 duplicate groups** (1,836 files). Kept the highest-quality successful run per group and safely archived **1,229 redundant files** to `results/_archive/duplicates_20261008/`.
- **Unmapped Names**: Purged synthetic test artifacts (`none`, `replace`, `replaceme_compressor`) and normalized 30+ legacy variant names directly to canonical keys.

## Coverage After Fix

- **Modules with Results**: 219 / 231 canonical modules (**94.81%**)
  - Attention modules covered: 147 / 159 (92.5%)
  - Compressor modules covered: 73 / 73 (100.0%)
- **Models**: 7 primary foundation model architectures across 1,367 run checkpoints.
- **Datasets**: 9 downstream NLP / LM benchmark datasets.

## Missing Modules (GPU Execution Required)

The following 14 modules require GPU/CUDA runners or specialized kernel environments:

1. `grouped_query_attention`
2. `adaptive_sparse_attn`
3. `x_attention`
4. `svd_linear_attn`
5. `simplessm`
6. `selective_mamba`
7. `hgrn2_gated_outer`
8. `hymba_transx_attn`
9. `fused_svd_kernel_attention`
10. `inject_triton_cosformer`
11. `inject_triton_svd_linear_attention`
12. `triton_cosformer_inject`
13. `triton_svd_linear_attention_inject`
14. `inject_xattention`

*Runner script ready*: `scripts/benchmark_missing_modules.sh`

## Next Steps

1. Run `scripts/benchmark_missing_modules.sh` on a GPU cluster node to evaluate the 14 remaining attention modules.
2. Ingest output metrics into `results/_MASTER_SUMMARY.csv` to hit 100% module coverage.
3. Final production release on PyPI.
