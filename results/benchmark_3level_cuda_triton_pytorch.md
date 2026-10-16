# 🚀 3-Level Execution Hierarchy Benchmark: Native CUDA C++ vs. Triton JIT vs. PyTorch

**Device:** NVIDIA GeForce RTX 4070 Laptop GPU (SM (8, 9))

| Kernel Operator | PyTorch Ref (ms) | Triton JIT (ms) | CUDA C++ (ms) | Speedup (CUDA vs. Py) | Speedup (CUDA vs. Triton) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `selective_scan` | 27.227 | 13.810 | 0.296 | **92.08x** | 46.71x |
| `sliding_window_attn` | 2.000 | 0.276 | 0.277 | **7.22x** | 1.00x |
| `block_sparse_attn` | 1.992 | 1.660 | 0.397 | **5.02x** | 4.18x |
| `w4a16_gemm` | 0.114 | N/A | 0.212 | **0.54x** | 1.00x |
| `fp8_gemm` | 0.215 | 0.074 | 0.372 | **0.58x** | 0.20x |
| `flash_gqa` | 0.208 | 0.034 | 0.084 | **2.49x** | 0.40x |
| `paged_attention` | 1.654 | N/A | 0.162 | **10.18x** | 1.00x |
| `moe_routing` | 361.454 | N/A | 0.013 | **28466.32x** | 1.00x |
| `fused_lowrank_gated_gemm` | 0.027 | 0.041 | 0.870 | **0.03x** | 0.05x |
| `chunked_delta_gate_scan` | 43.208 | 0.123 | 0.592 | **72.98x** | 0.21x |
| `hgrn2_fused_bwd_scan` | 5.686 | 0.026 | 0.013 | **422.60x** | 1.93x |
| `cosformer_linear_attention` | 0.111 | 0.025 | 0.059 | **1.88x** | 0.42x |
| `fused_norm_linear` | 0.014 | 0.021 | 0.117 | **0.12x** | 0.18x |
| `fused_rotary_emb` | 0.029 | N/A | 0.020 | **1.47x** | 1.00x |
| `kronecker_matmul` | 0.031 | 0.026 | 0.309 | **0.10x** | 0.08x |
| `lut_codebook_dequant_gemm` | 0.096 | 0.027 | 0.021 | **4.50x** | 1.26x |
| `linear_attention` | 40.532 | 0.262 | 0.242 | **167.81x** | 1.09x |
| `fused_cross_entropy` | 0.013 | 0.064 | 0.045 | **0.30x** | 1.43x |
