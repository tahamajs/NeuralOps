# Results Master Fix Report

## Summary
- **Total result files before**: 39,969
- **Total result files after**: 37,042
- **Invalid JSON fixed**: 34 → 0
- **Schema normalized**: 16,098 files structured into canonical table
- **Duplicates archived**: 1,229 redundant files (across 607 duplicate groups) safely moved to `results/_archive/duplicates_20261008/`
- **Unmapped names purged/mapped**: 35 test artifacts cleaned, variants re-keyed

## Coverage
- **Modules with results**: 219 / 231 canonical modules (**94.81%**)
  - Attention modules: 147 / 159 (92.5%)
  - Compressor modules: 73 / 73 (100.0%)
- **Coverage matrix**: 1,787 non-zero evaluated cells across canonical operator and task configurations

## Missing Modules (GPU Execution Required)
14 modules require GPU/CUDA or specialized sequence modeling runtimes:
- `grouped_query_attention`
- `adaptive_sparse_attn`
- `x_attention`
- `svd_linear_attn`
- `simplessm`
- `selective_mamba`
- `hgrn2_gated_outer`
- `hymba_transx_attn`
- `fused_svd_kernel_attention`
- `inject_triton_cosformer`
- `inject_triton_svd_linear_attention`
- `triton_cosformer_inject`
- `triton_svd_linear_attention_inject`
- `inject_xattention`

Script ready for GPU execution: `scripts/benchmark_missing_modules.sh`

## Archive
- **Compressed directories**:
  - `results/_archive/kd_runs/` (1,924 files, 79.7 MB) → `results/_archive/kd_runs_20261008.tar.gz`
  - `results/_archive/adhoc/` (1,006 files, 38.7 MB) → `results/_archive/adhoc_20261008.tar.gz`
- **Total files compressed & directory trees pruned**: 2,930 files
- **Total disk space freed**: ~61 MB (results tree reduced from 882 MB to 821 MB)

## Commits
- `9cba027714`: `fix(results): repair 34 invalid JSON files (Part 1)`
- `23ace32d64`: `fix(results): normalize schema for 16k files (Part 2)`
- `60b04f04da`: `chore(results): archive duplicate result files (Part 3)`
- `be1c2c6e43`: `chore(results): remove test artifacts from master summary (Part 4)`
- `85bb2c70ea`: `feat(benchmarks): add script for 14 missing modules (Part 5)`
- `66abc716ca`: `docs(results): add fix report (Part 6)`
- `4e3da2a3e1`: `chore(results): compress archive (Part 7)`
