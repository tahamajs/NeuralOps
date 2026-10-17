# 📊 Hardware Benchmarks of 3-Level Native CUDA C++ Operators

**Category Filter:** `cuda_kernel` | **Total Verified Records:** 36

| Module                     | Category    | Family                   | Setup / Dataset | Speedup X | Latency Triton Ms | Method                   | Ver (Rev)   | Git SHA  | Timestamp  |
|:--------------------------|-:-----------|-:------------------------|-:---------------|-:---------|-:-----------------|-:------------------------|-:-----------|-:--------|-:---------:|
| selective_scan             | cuda_kernel | native-cuda-acceleration | Standard        | 109.34    | 14.34             | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| sliding_window_attn        | cuda_kernel | native-cuda-acceleration | Standard        | 7.24      | 0.25              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| block_sparse_attn          | cuda_kernel | native-cuda-acceleration | Standard        | 4.98      | 1.53              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| w4a16_gemm                 | cuda_kernel | native-cuda-acceleration | Standard        | 0.39      | 0.00              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| fp8_gemm                   | cuda_kernel | native-cuda-acceleration | Standard        | 0.15      | 0.07              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| flash_gqa                  | cuda_kernel | native-cuda-acceleration | Standard        | 1.96      | 0.05              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| paged_attention            | cuda_kernel | native-cuda-acceleration | Standard        | 45.89     | 0.00              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| moe_routing                | cuda_kernel | native-cuda-acceleration | Standard        | 1336.96   | 0.00              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| fused_lowrank_gated_gemm   | cuda_kernel | native-cuda-acceleration | Standard        | 0.67      | 0.04              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| chunked_delta_gate_scan    | cuda_kernel | native-cuda-acceleration | Standard        | 82.40     | 0.13              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| hgrn2_fused_bwd_scan       | cuda_kernel | native-cuda-acceleration | Standard        | 436.29    | 0.03              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| cosformer_linear_attention | cuda_kernel | native-cuda-acceleration | Standard        | 1.41      | 0.06              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| fused_norm_linear          | cuda_kernel | native-cuda-acceleration | Standard        | 0.10      | 0.02              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| fused_rotary_emb           | cuda_kernel | native-cuda-acceleration | Standard        | 2.87      | 0.00              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| kronecker_matmul           | cuda_kernel | native-cuda-acceleration | Standard        | 1.78      | 0.03              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| lut_codebook_dequant_gemm  | cuda_kernel | native-cuda-acceleration | Standard        | 5.00      | 0.03              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| linear_attention           | cuda_kernel | native-cuda-acceleration | Standard        | 243.98    | 0.26              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| fused_cross_entropy        | cuda_kernel | native-cuda-acceleration | Standard        | 0.35      | 0.06              | 3-Level Native CUDA C++  | v1.3.7 (r1) | 2395ff7d | 2026-10-17 |
| selective_scan             | cuda_kernel | 3-level-native-cuda      | Standard        | 92.08     | 13.81             | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| sliding_window_attn        | cuda_kernel | 3-level-native-cuda      | Standard        | 7.22      | 0.28              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| block_sparse_attn          | cuda_kernel | 3-level-native-cuda      | Standard        | 5.02      | 1.66              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| w4a16_gemm                 | cuda_kernel | 3-level-native-cuda      | Standard        | 0.54      | 0.00              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| fp8_gemm                   | cuda_kernel | 3-level-native-cuda      | Standard        | 0.58      | 0.07              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| flash_gqa                  | cuda_kernel | 3-level-native-cuda      | Standard        | 2.49      | 0.03              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| paged_attention            | cuda_kernel | 3-level-native-cuda      | Standard        | 10.18     | 0.00              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| moe_routing                | cuda_kernel | 3-level-native-cuda      | Standard        | 28466.32  | 0.00              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| fused_lowrank_gated_gemm   | cuda_kernel | 3-level-native-cuda      | Standard        | 0.03      | 0.04              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| chunked_delta_gate_scan    | cuda_kernel | 3-level-native-cuda      | Standard        | 72.98     | 0.12              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| hgrn2_fused_bwd_scan       | cuda_kernel | 3-level-native-cuda      | Standard        | 422.60    | 0.03              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| cosformer_linear_attention | cuda_kernel | 3-level-native-cuda      | Standard        | 1.88      | 0.02              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| fused_norm_linear          | cuda_kernel | 3-level-native-cuda      | Standard        | 0.12      | 0.02              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| fused_rotary_emb           | cuda_kernel | 3-level-native-cuda      | Standard        | 1.47      | 0.00              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| kronecker_matmul           | cuda_kernel | 3-level-native-cuda      | Standard        | 0.10      | 0.03              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| lut_codebook_dequant_gemm  | cuda_kernel | 3-level-native-cuda      | Standard        | 4.50      | 0.03              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| linear_attention           | cuda_kernel | 3-level-native-cuda      | Standard        | 167.81    | 0.26              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
| fused_cross_entropy        | cuda_kernel | 3-level-native-cuda      | Standard        | 0.30      | 0.06              | Native CUDA C++ Kernel w | v1.3.7 (r1) | 9ed147b9 | 2026-10-16 |
