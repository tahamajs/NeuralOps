# NeuralOps Fallback Registry (v1.5.0)

This document is the authoritative registry of modules that intentionally utilize high-level Level-2 Triton JIT or Level-3 PyTorch routing, including the architectural justification, Triton entrypoint, and CUDA C++ kernel development status.

| Module | Classification Reason | Primary Acceleration Path | CUDA C++ Kernel Status |
|---|---|---|---|
| `flash_decoding` | Single-query decoding with Split-KV across sequence dimension $N_{kv}$ | `neuralops.core.triton_kernels.flash_decoding` | `EXPERIMENTAL` (Fused cooperative Split-KV in dev) |
| `cluster_attention` | Cluster centroids $K \ll N$ (e.g. $K \le 8$); cuBLAS Tensor Core $[N, K] \times [K, D]$ dominates memory-bound CUDA loops | `triton_cluster_attention` | `EXPERIMENTAL` (cuBLAS GEMM optimal for small $K$) |
| `nystrom_attention` | Landmark matrix decomposition $(Q_l P^\dagger) V_l$ computed via dual tiled Tensor Core contractions | `triton_nystrom_landmark` | `EXPERIMENTAL` (Tiled GEMM reference) |
| `linformer_attention` | Low-rank projected key/value sequence contraction $E K, F V$ via dual GEMM | `triton_linformer` | `EXPERIMENTAL` (Dual GEMM optimal) |
