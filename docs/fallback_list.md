# NeuralOps Fallback Registry & Known Bottlenecks Ledger

This ledger documents operations in NeuralOps that are currently forced to alternative dispatch implementations (Triton JIT or PyTorch SDPA) due to known kernel bottlenecks, algorithmic constraints, or hardware architectural limits.

---

## fused_norm_linear

- **Status:** `FALLBACK_ONLY` (CUDA kernel = `EXPERIMENTAL`, not dispatched)
- **Benchmark evidence:** 225–244× SLOWER than PyTorch reference
  (measured 2026-10-18, 9 configurations, RTX 4070 SM 8.9)
  - Fastest: B=8, S=512, H=4096 → 827.59 ms CUDA vs 3.39 ms PyTorch (244× slower)
  - Best: B=4, S=512, H=2048 → 106.06 ms CUDA vs 0.44 ms PyTorch (240× slower)
- **Root cause:**
  1. Serial `for(k=0..K)` loop per output element — no shared-memory tiling
  2. Uncoalesced memory access: `weight[n*K + k]` with stride = K
  3. `x_norm = (x[m*K+k] - mean) * rstd` recomputed on every iteration of the outer loops
- **Fix path:**
  Rewrite kernel with 2D shared-memory tiling staging `x` into `smem_x[BM][BK]` and `weight` into `smem_w[BN][BK]`. Use `act_scale_gemm.cu` as architecture template.
- **Routing:** Forced Triton (Level 2) → PyTorch (Level 3). Dispatch logic in `norm_linear_autograd.py` bypasses CUDA kernel.
- **Timeline:** v1.5.0 enhancement milestone. Not blocking v1.4.0 camera-ready thesis release.

## selective_scan

- **Status:** `CUDA_VERIFIED` (not fallback-only)
- **Evidence:** isolated RTX 4070 runs recorded speedups of 77.11x–81.79x
  with maximum absolute difference `3.0518e-05`.

## speculative_verify

- **Status:** `FALLBACK_ONLY` for performance certification
- **Evidence:** isolated RTX 4070 runs recorded speedups of 0.71x–0.73x
  with maximum absolute difference `0.0`.
- **Reason:** the native implementation is numerically exact but slower than
  the PyTorch reference at the certified shape; routing to the reference is
  preferable to claiming a CUDA speedup.

## entropy_pruning
- **Status:** TIMEOUT / FALLBACK_ONLY
- **Evidence:** exceeded the 120-second isolated benchmark limit.
- **Reason:** no completed CUDA result was available within the bounded run.

## logcumsumexp
- **Status:** TIMEOUT / FALLBACK_ONLY
- **Evidence:** exceeded the 120-second isolated benchmark limit.
- **Reason:** no completed CUDA result was available within the bounded run.

## samba_moe
- **Status:** TIMEOUT / FALLBACK_ONLY
- **Evidence:** exceeded the 120-second isolated benchmark limit.
- **Reason:** no completed CUDA result was available within the bounded run.

## sparse_attention
- **Status:** TIMEOUT / FALLBACK_ONLY
- **Evidence:** Group A was stopped after the 120-second module timeout.
- **Reason:** no completed CUDA result was available within the bounded run.
