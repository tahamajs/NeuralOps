# 🚀 3-Level Execution Hierarchy Benchmark: Native CUDA C++ vs. Triton JIT vs. PyTorch

**Device:** NVIDIA GeForce RTX 4070 Laptop GPU (SM (8, 9))

| Kernel Operator | PyTorch Ref (ms) | Triton JIT (ms) | CUDA C++ (ms) | Speedup (CUDA vs. Py) | Speedup (CUDA vs. Triton) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `selective_scan` | 29.936 | 14.339 | 0.274 | **109.34x** | 52.37x |
| `sliding_window_attn` | 1.826 | 0.252 | 0.252 | **7.24x** | 1.00x |
| `block_sparse_attn` | 1.819 | 1.529 | 0.365 | **4.98x** | 4.19x |
| `w4a16_gemm` | 0.105 | N/A | 0.267 | **0.39x** | 1.00x |
| `fp8_gemm` | 0.078 | 0.068 | 0.509 | **0.15x** | 0.13x |
| `flash_gqa` | 0.192 | 0.054 | 0.098 | **1.96x** | 0.56x |
| `paged_attention` | 1.096 | N/A | 0.024 | **45.89x** | 1.00x |
| `moe_routing` | 29.505 | N/A | 0.022 | **1336.96x** | 1.00x |
| `fused_lowrank_gated_gemm` | 0.026 | 0.040 | 0.038 | **0.67x** | 1.04x |
| `chunked_delta_gate_scan` | 47.587 | 0.126 | 0.578 | **82.40x** | 0.22x |
| `hgrn2_fused_bwd_scan` | 6.209 | 0.027 | 0.014 | **436.29x** | 1.87x |
| `cosformer_linear_attention` | 0.120 | 0.064 | 0.085 | **1.41x** | 0.76x |
| `fused_norm_linear` | 0.014 | 0.020 | 0.139 | **0.10x** | 0.15x |
| `fused_rotary_emb` | 0.029 | N/A | 0.010 | **2.87x** | 1.00x |
| `kronecker_matmul` | 0.030 | 0.025 | 0.017 | **1.78x** | 1.49x |
| `lut_codebook_dequant_gemm` | 0.096 | 0.026 | 0.019 | **5.00x** | 1.35x |
| `linear_attention` | 40.823 | 0.257 | 0.167 | **243.98x** | 1.53x |
| `fused_cross_entropy` | 0.014 | 0.063 | 0.040 | **0.35x** | 1.60x |
