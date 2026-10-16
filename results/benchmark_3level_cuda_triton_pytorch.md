# 🚀 3-Level Execution Hierarchy Benchmark: Native CUDA C++ vs. Triton JIT vs. PyTorch

**Device:** NVIDIA GeForce RTX 4070 Laptop GPU (SM (8, 9))

| Kernel Operator | PyTorch Ref (ms) | Triton JIT (ms) | CUDA C++ (ms) | Speedup (CUDA vs. Py) | Speedup (CUDA vs. Triton) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `selective_scan` | 27.425 | 13.683 | 0.274 | **100.05x** | 49.91x |
| `sliding_window_attn` | 1.823 | 0.251 | 0.251 | **7.26x** | 1.00x |
| `block_sparse_attn` | 1.816 | 1.528 | 0.364 | **4.99x** | 4.19x |
| `w4a16_gemm` | 0.104 | N/A | 0.195 | **0.53x** | 1.00x |
| `fp8_gemm` | 0.078 | 0.068 | 0.342 | **0.23x** | 0.20x |
| `flash_gqa` | 0.191 | 0.055 | 0.097 | **1.97x** | 0.57x |
| `paged_attention` | 1.091 | N/A | 0.040 | **27.22x** | 1.00x |
| `moe_routing` | 28.606 | N/A | 0.015 | **1905.56x** | 1.00x |
