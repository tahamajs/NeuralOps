# 🚀 3-Level Execution Hierarchy Benchmark: Native CUDA C++ vs. Triton JIT vs. PyTorch

**Device:** NVIDIA GeForce RTX 4070 Laptop GPU (SM (8, 9))

| Kernel Operator | PyTorch Ref (ms) | Triton JIT (ms) | CUDA C++ (ms) | Speedup (CUDA vs. Py) | Speedup (CUDA vs. Triton) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `selective_scan` | 27.095 | 13.648 | 0.273 | **99.10x** | 49.92x |
| `sliding_window_attn` | 1.822 | 0.252 | 0.251 | **7.26x** | 1.00x |
| `block_sparse_attn` | 1.816 | 1.526 | 0.364 | **4.98x** | 4.19x |
| `w4a16_gemm` | 0.104 | N/A | 0.266 | **0.39x** | 1.00x |
| `fp8_gemm` | 0.078 | 0.068 | 0.507 | **0.15x** | 0.13x |
| `flash_gqa` | 0.192 | 0.032 | 0.079 | **2.43x** | 0.41x |
| `paged_attention` | 1.083 | N/A | 0.024 | **45.82x** | 1.00x |
| `moe_routing` | 29.596 | N/A | 0.022 | **1340.59x** | 1.00x |
| `fused_lowrank_gated_gemm` | 0.025 | 0.040 | 0.038 | **0.65x** | 1.06x |
| `chunked_delta_gate_scan` | 43.506 | 0.127 | 0.466 | **93.38x** | 0.27x |
| `hgrn2_fused_bwd_scan` | 5.695 | 0.031 | 0.013 | **435.51x** | 2.35x |
| `cosformer_linear_attention` | 0.118 | 0.027 | 0.106 | **1.12x** | 0.26x |
| `fused_norm_linear` | 0.015 | 0.019 | 0.195 | **0.08x** | 0.10x |
| `fused_rotary_emb` | 0.028 | N/A | 0.008 | **3.43x** | 1.00x |
| `kronecker_matmul` | 0.030 | 0.026 | 0.017 | **1.79x** | 1.51x |
| `lut_codebook_dequant_gemm` | 0.084 | 0.025 | 0.019 | **4.38x** | 1.31x |
| `linear_attention` | 41.680 | 0.266 | 0.263 | **158.67x** | 1.01x |
| `fused_cross_entropy` | 0.012 | 0.065 | 0.037 | **0.31x** | 1.75x |

*Note: For `chunked_delta_gate_scan`, Triton JIT achieves 343.66x speedup over PyTorch (0.127 ms vs 43.506 ms).*
