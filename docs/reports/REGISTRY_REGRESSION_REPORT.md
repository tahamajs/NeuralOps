# Registry Regression Report

**Date:** 2026-10-08  
**Subject:** Attention Registry Count Regression (159 → 145)  
**Status:** Diagnosed & Solution Ready

---

## 1. Executive Summary

During prior cleanup passes aimed at eliminating redundant registry decorators and consolidating aliases, 14 canonical alias names were moved out of direct `@ATTENTION_REGISTRY.register` decorator calls into `deprecated_aliases.json`. 

This reduced the live in-memory registry key count from **159** to **145**. Meanwhile, the compressor count remained completely intact at **73**.

## 2. Root Cause Analysis

### Commits Responsible:
- Commit `c49720421c` maintained all **159** attention modules + **73** compressors in `ATTENTION_REGISTRY`.
- Subsequent cleanup commits (`3a3b87774a` and related PRs documented in `docs/DUPLICATE_MODULE_AUDIT.md` and `docs/INJECTOR_MODULE_AUDIT.md`) removed duplicate `@ATTENTION_REGISTRY.register` decorators from class headers, treating them as pure aliases.
- Because `ATTENTION_REGISTRY.keys()` counts registered decorator keys directly rather than including aliases in `deprecated_aliases.json`, the visible key count dropped to 145.

### File Changes Involved:
Removal of secondary `@ATTENTION_REGISTRY.register(...)` decorators on existing class definitions across:
- `neuralops/attentions/svd/adaptive_svd_attention.py` (`adaptive_svd`)
- `neuralops/attentions/sparse/adaptive_sparse_attention.py` (`adaptive_sparse_attn`)
- `neuralops/attentions/svd/asvd_attention.py` (`asvd`)
- `neuralops/attentions/svd/factorised_matrix_state.py` (`fms`)
- `neuralops/attentions/dense/full_attention.py` (`full`)
- `neuralops/attentions/modern/gqa_attention.py` (`grouped_query_attention`)
- `neuralops/attentions/gated/hgrn2_gated_outer.py` (`hgrn2_gated_outer`)
- `neuralops/attentions/ssm/hymba_transx.py` (`hymba_transx_attn`)
- `neuralops/attentions/sparse/nsa_attention.py` (`nsa`)
- `neuralops/attentions/ssm/selective_ssm.py` (`simplessm`)
- `neuralops/attentions/svd/svd_linear_attention.py` (`svd_linear_attn`)
- `neuralops/attentions/modern/vig_attention.py` (`vig`)
- `neuralops/attentions/modern/wgqa_attention.py` (`wgqa`)
- `neuralops/attentions/fast_kernels/xattention.py` (`x_attention`)

---

## 3. The 14 Missing Modules and Their Targets

| # | Missing Key | Target Implementation Class | File Path |
|:---:|:---|:---|:---|
| 1 | `grouped_query_attention` | `GroupedQueryAttention` | `neuralops/attentions/modern/gqa_attention.py` |
| 2 | `adaptive_sparse_attn` | `AdaptiveSparseAttention` | `neuralops/attentions/sparse/adaptive_sparse_attention.py` |
| 3 | `x_attention` | `XAttention` | `neuralops/attentions/fast_kernels/xattention.py` |
| 4 | `svd_linear_attn` | `SVDLinearAttention` | `neuralops/attentions/svd/svd_linear_attention.py` |
| 5 | `simplessm` | `SelectiveSSM` | `neuralops/attentions/ssm/selective_ssm.py` |
| 6 | `hgrn2_gated_outer` | `HGRN2GatedOuterBlock` | `neuralops/attentions/gated/hgrn2_gated_outer.py` |
| 7 | `hymba_transx_attn` | `HymbaTransXBlock` | `neuralops/attentions/ssm/hymba_transx.py` |
| 8 | `adaptive_svd` | `AdaptiveSVDAttention` | `neuralops/attentions/svd/adaptive_svd_attention.py` |
| 9 | `asvd` | `ASVDAttention` | `neuralops/attentions/svd/asvd_attention.py` |
| 10 | `fms` | `FactorisedMatrixState` | `neuralops/attentions/svd/factorised_matrix_state.py` |
| 11 | `full` | `FullAttention` | `neuralops/attentions/dense/full_attention.py` |
| 12 | `nsa` | `NativeSparseAttention` | `neuralops/attentions/sparse/nsa_attention.py` |
| 13 | `vig` | `ViGAttention` | `neuralops/attentions/modern/vig_attention.py` |
| 14 | `wgqa` | `WeightedGroupedQueryAttention` | `neuralops/attentions/modern/wgqa_attention.py` |

---

## 4. Proposed Fix

Restore registration for these 14 module names directly onto the underlying classes via `@ATTENTION_REGISTRY.register` (or inside the module init), ensuring that:
1. `len(ATTENTION_REGISTRY.keys()) == 159`
2. `len(COMPRESSOR_REGISTRY.keys()) == 73`
3. Total active modules: **232 (159 attention + 73 compressor)**
4. All existing tests, downstream pipelines, and lookup keys function without warning or regression.
