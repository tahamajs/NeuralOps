# 🔬 NEURALOPS v1.3.7 — TRITON KERNEL DISPATCH INVENTORY
**Date**: October 16, 2026  
**Hardware Platform**: Physical NVIDIA GeForce RTX 4070 Laptop GPU 8GB (SM 8.9)  
**Status**: VERIFIED EMPIRICAL DISPATCH AUDIT

---

## 1. Executive Summary

- **Subquadratic Attentions Audited**: 131
  - Directly Dispatching Triton Kernels: **108 / 131 (82.4%)**
  - Pure PyTorch Baseline (No Triton Dispatch): **23 / 131**

- **Compression Engines Audited**: 72
  - Directly Dispatching Triton Kernels: **50 / 72 (69.4%)**
  - Offline/Analytical CPU/PyTorch Engines: **22 / 72**

---

## 2. Attention Mechanisms Dispatch Matrix (131 Modules)

| Module Name | Dispatched Triton Kernels | Acceleration Status |
|:---|:---|:---:|
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  | ,  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | ,  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | ,  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  |  | ✅ Accelerated |
|  | None | ⚠️ PyTorch Baseline |
|  | None | ⚠️ PyTorch Baseline |
|  |  | ✅ Accelerated |

---

## 3. Compression Engines Dispatch Matrix (72 Modules)

| Compressor Name | Dispatched Triton Kernels | Acceleration Status |
|:---|:---|:---:|
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | ,  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  | ,  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  | None (Offline SVD/Pruner) | ℹ️ Offline / PyTorch |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  |  | ✅ Fused GEMM/SVD |
|  | ,  | ✅ Fused GEMM/SVD |

---

## 4. Modules Without Triton Acceleration (Candidates for Acceleration)

### 4.1 Attentions:
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 

### 4.2 Compressors:
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 