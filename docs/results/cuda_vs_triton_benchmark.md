# CUDA C++ vs PyTorch Reference — Measured Hot-Path Benchmarks

| Module | CUDA (ms) | PyTorch (ms) | Speedup vs PyTorch |
|:---|:---:|:---:|:---:|
| **paged_attention** | 1.624 | 3.337 | **2.06×** |
| **selective_scan** | 0.056 | 6.995 | **125.95×** |
| **w4a16_gemm** | 0.015 | 0.070 | **4.56×** |
| **fp8_gemm** | 1.447 | 1.429 | **0.99×** |
| **flash_gqa** | 1.627 | 0.035 | **0.02×** |
| **moe_routing** | 0.021 | 0.020 | **0.97×** |
| **sparse_attention** | 0.411 | 0.035 | **0.08×** |
| **sliding_window** | 0.200 | 0.103 | **0.51×** |
| **paged_kv_cache** | 0.003 | 5.510 | **1963.82×** |
