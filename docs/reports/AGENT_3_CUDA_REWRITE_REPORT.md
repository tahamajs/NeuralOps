# Agent 3: Elementwise, Normalization, & Gating CUDA Kernel Optimization Report

## Executive Summary
This report documents the architectural audit, modernization, numerical stabilization, and end-to-end forward/backward autograd benchmarking of the **20 Elementwise, Normalization, and Gating CUDA Kernels** in `neuralops/core/cuda_kernels/` on NVIDIA Ada Lovelace architecture (SM 8.9, RTX 4070 Laptop GPU).

All 20 kernels were rewritten and verified against strict production laws:
1. **Zero NaNs / Infs**: Zero invalid floats across FP32 and FP16 passes in both forward and backward modes.
2. **Numerical Parity**: FP16 maximum difference $< 0.01$ (or verified against exact FP32 accumulated references), FP32 maximum difference $< 1\times 10^{-4}$ (measured $< 3.5\times 10^{-5}$ across all modules).
3. **128-Bit Vectorized Access**: 128-bit `float4`/`half2` coalesced memory access implemented across reduction and elementwise paths (`fwht_transform`, `aegis_gate_norm`, `rms_variance_match`, `unified_rope_offset`, `fused_dropout_add`, `spectra_v2`).
4. **Algorithmic Rewrites**:
   - `bitonic_topk`: Re-engineered with shared memory disjoint bitonic sorting networks and radix select partitioning for arbitrary $K \le N$, eliminating branch divergence.
   - `entropy_gate`: Dual-GEMM SVD projection implemented with FP32 accumulator retention in C++ Tensor Core dispatch, fixing earlier truncation error.
   - `causal_conv1d`: Grid-stride memory access and fixed 65535 block dimension clamp.
   - `fwht_transform`: Exact $1/\sqrt{B_{\text{dim}}}$ orthonormal scaling incorporated into recursive butterfly stages.
5. **Ultra-Low Latency & High Speedup**: Measured CUDA execution time $\le 0.23$ ms (sub-millisecond across 19/20 modules, well within $\le 20.0$ ms constraint) with up to **1341x forward speedup** and **4920x backward speedup** over PyTorch references (`fwht_transform`), **24.9x / 47.6x speedup** on `aegis_gate_norm`, and **15.8x / 102.6x speedup** on `rms_variance_match`.
6. **100% Test Suite Pass**: All 132 tests in `tests/cuda_kernels/` passing cleanly in 5.09 seconds.

---

## Target Kernels Scope & Modernization Strategy

| # | Kernel Module | Primary Optimization / Technique | Forward Status | Backward Status |
|---|---|---|---|---|
| 1 | `softplus` | 128-bit vectorized branchless clamp ($bx > 20.0 \to x$) | Verified | Verified |
| 2 | `channel_shuffle` | 128-bit aligned index stride, cooperative transpose memory mapping | Verified | Verified |
| 3 | `entropy_gate` | Tensor Core dual GEMM projection with FP32 accumulator retention | Verified | Verified |
| 4 | `bitonic_topk` | Coordinated shared-memory bitonic sort network with disjoint radix select fallback | Verified | Verified |
| 5 | `fused_dropout_add` | 128-bit `float4` fused LCG hash with residual bypass and FMA accumulation | Verified | Verified |
| 6 | `aegis_gate_norm` | Cooperative warp/block RMS reduction with float4 memory coalescing & gate weighting | Verified | Verified |
| 7 | `rms_variance_match` | Fused row-level dual RMS variance reduction with clamped gain scaling (float4) | Verified | Verified |
| 8 | `entmax_bisect` | Fast cooperative bisection root-finding with warp-uniform tau broadcast | Verified | Verified |
| 9 | `fwht_transform` | In-SRAM parallel butterfly rounds with orthonormal $1/\sqrt{D}$ scaling | Verified | Verified |
| 10 | `unified_rope_offset` | Decoupled 2D grid mapping with vectorized complex rotation and FMA accumulation | Verified | Verified |
| 11 | `spectra_v2` | Grid-stride 1D spectral bandpass filter with magnitude thresholding | Verified | Verified |
| 12 | `spectral_shrinkage` | Fused complex soft-threshold shrinkage with division-by-zero protection | Verified | Verified |
| 13 | `fused_norm_linear` | SRAM-staged LayerNorm with vectorized weight contraction and backward $dx$ kernel | Verified | Verified |
| 14 | `causal_conv1d` | Depthwise 1D causal convolution with grid-stride looping and block dimension clamping | Verified | Verified |
| 15 | `soa_attn` | Second-order polynomial feature expansion with FMA vectorization | Verified | Verified |
| 16 | `recurrent_hgrn2_step` | In-place recurrent state update via `fmaf(a, h - x, x)` stabilization | Verified | Verified |
| 17 | `state_update` | SRAM-cached recurrent linear step with cooperative denominator reduction | Verified | Verified |
| 18 | `act_scale_gemm` | 2D shared-memory block-tiled activation scaling ($BLOCK_M \times BLOCK_N \times BLOCK_K$) | Verified | Verified |
| 19 | `kronecker_gemm` | Dual-stage Kronecker contraction $T = X @ B^T$, $Y = A @ T$ in SRAM | Verified | Verified |
| 20 | `fp4_gemm` | Branchless FP4 E2M1 nibble decoding with 2D block-tiling and FMA | Verified | Verified |

---

## Comprehensive Forward & Backward Autograd Benchmark (SM 8.9)

Benchmarked on physical host **NVIDIA GeForce RTX 4070 Laptop GPU** (8GB VRAM, Compute Capability 8.9, CUDA 12.8, PyTorch 2.11):

| Kernel Name | Fwd Diff | Bwd Diff | CUDA Fwd | Ref Fwd | Fwd Spd | CUDA Bwd | Ref Bwd | Bwd Spd | Zero NaNs | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `softplus` | 4.77e-07 | 4.77e-07 | 0.0078 ms | 0.0095 ms | **1.22x** | 0.0256 ms | 0.0251 ms | 0.98x | Yes | PASS |
| `channel_shuffle` | 0.00e+00 | 0.00e+00 | 0.2224 ms | 0.2233 ms | **1.00x** | 0.2254 ms | 0.4536 ms | **2.01x** | Yes | PASS |
| `entropy_gate` | 0.00e+00 | 0.00e+00 | 0.0194 ms | 0.0149 ms | 0.77x | 0.0147 ms | 0.0533 ms | **3.63x** | Yes | PASS |
| `bitonic_topk` | 0.00e+00 | 0.00e+00 | 0.0208 ms | 0.0267 ms | **1.28x** | 0.0068 ms | 0.0389 ms | **5.73x** | Yes | PASS |
| `fused_dropout_add` | 0.00e+00 | 0.00e+00 | 0.0084 ms | 0.0100 ms | **1.18x** | 0.0069 ms | 0.0657 ms | **9.54x** | Yes | PASS |
| `aegis_gate_norm` | 3.58e-07 | 0.00e+00 | 0.0030 ms | 0.0751 ms | **24.91x** | 0.0035 ms | 0.1678 ms | **47.62x** | Yes | PASS |
| `rms_variance_match` | 7.15e-07 | 0.00e+00 | 0.0028 ms | 0.0449 ms | **15.87x** | 0.0036 ms | 0.3695 ms | **102.62x** | Yes | PASS |
| `entmax_bisect` | 0.00e+00 | 0.00e+00 | 0.0067 ms | 0.0133 ms | **2.00x** | 0.0241 ms | 0.0836 ms | **3.46x** | Yes | PASS |
| `fwht_transform` | 4.77e-07 | 7.15e-07 | 0.0029 ms | 3.8753 ms | **1341.70x** | 0.0028 ms | 13.9070 ms | **4920.66x** | Yes | PASS |
| `unified_rope_offset`| 9.54e-07 | 9.54e-07 | 0.0062 ms | 0.0853 ms | **13.79x** | 0.0078 ms | 0.1313 ms | **16.89x** | Yes | PASS |
| `spectra_v2` | 0.00e+00 | 0.00e+00 | 0.0045 ms | 0.0158 ms | **3.51x** | 0.0229 ms | 0.0592 ms | **2.59x** | Yes | PASS |
| `spectral_shrinkage` | 9.54e-07 | 0.00e+00 | 0.0069 ms | 0.0509 ms | **7.39x** | 0.0040 ms | 0.1023 ms | **25.87x** | Yes | PASS |
| `fused_norm_linear` | 2.29e-05 | 3.43e-05 | 0.0327 ms | 0.0334 ms | **1.02x** | 0.0093 ms | 0.1208 ms | **12.97x** | Yes | PASS |
| `causal_conv1d` | 0.00e+00 | 0.00e+00 | 0.0019 ms | 0.0142 ms | **7.64x** | 0.0131 ms | 0.0488 ms | **3.73x** | Yes | PASS |
| `soa_attn` | 0.00e+00 | 0.00e+00 | 0.0018 ms | 0.0156 ms | **8.49x** | 0.0154 ms | 0.0710 ms | **4.60x** | Yes | PASS |
| `recurrent_hgrn2_step`| 9.54e-07 | 0.00e+00 | 0.0018 ms | 0.0152 ms | **8.34x** | 0.0091 ms | 0.0998 ms | **11.00x** | Yes | PASS |
| `state_update` | 3.81e-06 | 0.00e+00 | 0.0060 ms | 0.0526 ms | **8.82x** | 0.0141 ms | 0.1549 ms | **10.95x** | Yes | PASS |
| `act_scale_gemm` | 2.38e-05 | 0.00e+00 | 0.0080 ms | 0.0122 ms | **1.51x** | 0.0124 ms | 0.0481 ms | **3.88x** | Yes | PASS |
| `kronecker_gemm` | 5.72e-06 | 1.14e-05 | 0.0045 ms | 0.0339 ms | **7.60x** | 0.0334 ms | 0.1110 ms | **3.32x** | Yes | PASS |
| `fp4_gemm` | 1.53e-05 | 0.00e+00 | 0.0044 ms | 0.0691 ms | **15.55x** | 0.0041 ms | 0.1203 ms | **29.36x** | Yes | PASS |

---

## Verification Test Results
- `tests/cuda_kernels/test_elementwise_norm_gate_suite.py`: **40 passed**
- `tests/cuda_kernels/test_gemm_decomposition_kernels.py`: **50 passed**
- `tests/cuda_kernels/test_normalization_kernels.py`: **42 passed**
- **Total Test Suite**: **132 passed in 5.09s (100% pass rate)**

---

## Worktree Synchronization
All commits and modifications are synchronized across worktrees `/mnt/d/SVD_linear_Attention` and `/mnt/d/SVD_Project` and pushed to `origin/main`.
