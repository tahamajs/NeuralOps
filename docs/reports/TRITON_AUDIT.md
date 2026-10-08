# Triton Kernel Comprehensive Audit Report

**Date**: 2026-10-09  
**Branch**: `ablation`  
**Framework**: NeuralOps Triton Accelerators (`neuralops/core/triton_kernels/`)

---

## 1. Executive Summary

This report documents the architectural audit, compatibility analysis, and CPU fallback mechanisms across all Triton kernels in NeuralOps.

- **Total Kernel Files**: 21
- **Hardware Targets**: NVIDIA GPU (CUDA via Triton JIT) with pure PyTorch CPU/MPS/ROCm fallback pathways.
- **Triton Availability**: Evaluated via `torch.cuda.is_available()` and dynamic import checks in `neuralops.core.triton_kernels._common` and `_types`.

---

## 2. Kernel Inventory & Status

| Kernel Module | Lines | Triton JIT Kernel | Forward (`fwd`) | Backward (`bwd`) | `is_supported` Check | PyTorch Fallback |
|---|---|---|---|---|---|---|
| `_common.py` | 615 | ❌ (Helper/Ref) | N/A | N/A | N/A | ✅ Comprehensive fallback ops |
| `_types.py` | 452 | ❌ (Type/Helper) | N/A | N/A | N/A | ✅ Fallback definitions & dispatch |
| `autotuner.py` | 58 | ❌ (Config) | N/A | N/A | N/A | ✅ Architecture heuristic tuner |
| `block_sparse.py` | 44 | ❌ (Layout) | N/A | N/A | N/A | ✅ Block sparsity mask utilities |
| `elementwise.py` | 52 | ✅ Fused pointwise | N/A | N/A | N/A | ✅ Fallback via torch operations |
| `flash_decoding.py` | 91 | ✅ Split-K Decoding | N/A | N/A | N/A | ✅ Standard torch decode fallback |
| `fms_scan.py` | 545 | ✅ Fast Memory Scan | ✅ | ✅ | ❌ | ✅ Pure torch iterative scan |
| `fused_rope.py` | 60 | ✅ Rotary Embeddings | N/A | N/A | N/A | ✅ `apply_rotary_emb` fallback |
| `gdn_scan.py` | 503 | ✅ Gated Decay Net | ✅ | ✅ | ❌ | ✅ Torch recursive scan |
| `gla_scan.py` | 482 | ✅ Gated Linear Attn | ✅ | ✅ | ❌ | ✅ Torch chunked scan |
| `gqa.py` | 166 | ✅ Grouped Query | ✅ | ✅ | ✅ | ✅ Pure PyTorch repeat & bmm |
| `hgrn2_scan.py` | 254 | ✅ HGRN2 Recurrence | ✅ | ✅ | ❌ | ✅ Torch sequential/parallel scan |
| `linear_attn.py` | 451 | ✅ Subquadratic Attn | ✅ | ✅ | ✅ | ✅ Chunk-based PyTorch linear attention |
| `low_rank.py` | 261 | ✅ Dual GEMM Bottleneck| ✅ | ✅ | ✅ | ✅ `low_rank_pytorch_fallback` |
| `paged_attention.py` | 96 | ✅ Paged KV Cache | N/A | N/A | N/A | ✅ Paged cache gather & attention |
| `selective_scan.py` | 437 | ✅ Mamba SSM Scan | ✅ | ✅ | ❌ | ✅ PyTorch SSM selective scan |
| `sliding_window.py` | 120 | ✅ Banded Local Window| N/A | N/A | N/A | ✅ Masked softmax local window |
| `softplus.py` | 59 | ✅ Numerically stable | N/A | N/A | N/A | ✅ `F.softplus` fallback |
| `sparse_attn.py` | 291 | ✅ Block-Sparse Attn | ✅ | ✅ | ✅ | ✅ `sparse_attn_pytorch_fallback` |
| `state_update.py` | 309 | ✅ Recurrent State Step| ✅ | ✅ | ✅ | ✅ Pure torch state step |
| `varlen_attention.py`| 51 | ❌ (Padding-free) | N/A | N/A | N/A | ✅ Cumulative sequence indexing |

---

## 3. Fallback Integrity & Safety

1. **CPU/MPS Execution**: When CUDA is unavailable or when tensors reside on CPU/MPS, `is_supported` or `check_triton_suitability` transparently directs computation to numerical reference implementations.
2. **Autograd Autodiff**: For all fused forward kernels with autograd bindings (`_Triton*AutogradFunction`), analytical backward kernels are paired with exact vector-Jacobian products.
3. **Registry Compatibility**: Models employing subquadratic linear attention (`svd_linear_attention`, `gdn`, `gla`, `hgrn2`) seamlessly fall back to non-Triton PyTorch execution when deployed in non-CUDA environments.
