# Triton Kernels Test Audit & Verification Report

**Date:** October 2026  
**Scope:** `neuralops/core/triton_kernels/` (all 16 modules/entrypoints)  
**Status:** All 16 kernels audited, tested, verified on CPU with transparent fallbacks and GPU guards.

---

## 1. Test Coverage Matrix

| Kernel File | Location | Test File | Test Count | CPU Pass | GPU Guard | Status |
|---|---|---|---|---|---|---|
| `_common.py` | `neuralops/core/triton_kernels/` | `test_triton_kernels_suite.py` | 6 | YES | YES | Fully Covered |
| `softplus.py` | `neuralops/core/triton_kernels/` | `test_triton_softplus.py` | 13 | YES | YES | Fully Covered |
| `gqa.py` | `neuralops/core/triton_kernels/` | `test_triton_gqa.py` | 27 | YES | YES | Fully Covered |
| `state_update.py` | `neuralops/core/triton_kernels/` | `test_triton_state_update.py` | 35 | YES | YES | Fully Covered |
| `low_rank.py` | `neuralops/core/triton_kernels/` | `test_triton_low_rank.py` | 31 | YES | YES | Fully Covered |
| `linear_attn.py` | `neuralops/core/triton_kernels/` | `test_triton_linear_attn.py` | 53 | YES | YES | Fully Covered |
| `sparse_attn.py` | `neuralops/core/triton_kernels/` | `test_triton_sparse_attn.py` | 21 | YES | YES | Fully Covered |
| `fms_scan.py` | `neuralops/core/triton_kernels/` | `test_triton_fms_scan.py` | 12 | YES | YES | Fully Covered |
| `gdn_scan.py` | `neuralops/core/triton_kernels/` | `test_triton_gdn_scan.py` | 12 | YES | YES | Fully Covered |
| `gla_scan.py` | `neuralops/core/triton_kernels/` | `test_triton_gla_scan.py` | 12 | YES | YES | Fully Covered |
| `hgrn2_scan.py` | `neuralops/core/triton_kernels/` | `test_triton_hgrn2_scan.py` | 19 | YES | YES | Fully Covered |
| `selective_scan.py` | `neuralops/core/triton_kernels/` | `test_triton_selective_scan.py` | 8 | YES | YES | Fully Covered |
| `triton_cosformer.py` | `neuralops/attentions/linear/` | `test_triton_cosformer_comprehensive.py` | 7 | YES | YES | Fully Covered |
| `triton_linear_attn.py` | `neuralops/attentions/linear/` | `test_triton_linear_attn.py` | (Bridge) | YES | YES | Fully Covered |
| `triton_sparse_attn.py` | `neuralops/attentions/sparse/` | `test_triton_sparse_attn.py` | (Bridge) | YES | YES | Fully Covered |
| `triton_svd_linear_attention.py` | `neuralops/attentions/svd/` | `test_triton_svd_linear_attention_comprehensive.py` | 8 | YES | YES | Fully Covered |
| **All Kernels Integration** | `neuralops/core/triton_kernels/` | `test_triton_all_kernels.py` | 3 | YES | YES | Fully Covered |

---

## 2. Test Execution Details

- **Total Test Cases:** 267+ parametrized tests across 14 test files.
- **CPU Execution:** All tests succeed without crashes, NaNs, or memory leaks using pure PyTorch references and fallbacks.
- **CUDA/Triton Execution:** When running on a machine with CUDA and Triton installed, JIT compiled kernels run numerical comparison against reference implementations with strict precision tolerances (`1e-4` for fp32, `1e-2` for fp16/bf16).
- **Continuous Integration:** Automated via `.github/workflows/test-triton.yml` on push and PR to `main` and `ablation`.

---

## 3. Summary of Implementations

1. **Batch 1 (Core Operations):**
   - Softplus, GQA, State Update, Low-Rank Projections, Linear Attention, Sparse Attention.
2. **Batch 2 (SSM / Recurrent Models):**
   - Factorised Matrix State (`fms_scan`), Gated DeltaNet (`gdn_scan`), Gated Linear Attention (`gla_scan`), HGRN2 (`hgrn2_scan`), Mamba Selective Scan (`selective_scan`).
3. **Batch 3 (Attention & Integration):**
   - Cosformer Attention, SVD Linear Attention, and All-Kernels integration suite.
