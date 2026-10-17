# 📊 Hardware Benchmarks of Triton JIT Accelerated Kernels

**Category Filter:** `triton_kernel` | **Total Verified Records:** 39

| Module                                 | Category      | Family                   | Setup / Dataset | Speedup X | Latency Triton Ms | Peak Vram Mb | Method                   | Ver (Rev)   | Git SHA  | Timestamp  |
|:--------------------------------------|-:-------------|-:------------------------|-:---------------|-:---------|-:-----------------|-:------------|-:------------------------|-:-----------|-:--------|-:---------:|
| triton_fused_lowrank_gated_gemm        | triton_kernel | sram-fused-kernel        | Standard        | 1.23      | 0.04              | 16.63        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_fused_lowrank_sparse_gemm       | triton_kernel | sram-fused-kernel        | Standard        | 0.59      | 10.46             | 33.99        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_strided_tt_linear               | triton_kernel | sram-fused-kernel        | Standard        | 1.07      | 0.32              | 50.09        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_fused_hadamard_w4a16_gemm       | triton_kernel | sram-fused-kernel        | Standard        | 1.62      | 0.04              | 34.74        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_lut_codebook_dequant_gemm       | triton_kernel | sram-fused-kernel        | Standard        | 5.20      | 0.03              | 34.74        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_fused_kronecker_matmul          | triton_kernel | sram-fused-kernel        | Standard        | 1.89      | 0.03              | 42.63        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_chunked_delta_gate_scan         | triton_kernel | sram-fused-kernel        | Standard        | 743.23    | 0.12              | 39.65        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_cosformer_linear_attention      | triton_kernel | sram-fused-kernel        | Standard        | 4.09      | 0.05              | 41.40        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_block_sparse_streaming_attn     | triton_kernel | sram-fused-kernel        | Standard        | 275.87    | 0.03              | 41.90        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_hgrn2_fused_bwd_scan            | triton_kernel | sram-fused-kernel        | Standard        | 214.04    | 0.03              | 43.15        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_spectral_entropy_svd_gate       | triton_kernel | sram-fused-kernel        | Standard        | 0.30      | 0.12              | 50.90        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_fused_act_scale_quant_gemm      | triton_kernel | sram-fused-kernel        | Standard        | 1.12      | 0.03              | 48.41        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| triton_fused_latent_kv_cache_attention | triton_kernel | sram-fused-kernel        | Standard        | 1.61      | 0.04              | 48.57        | Compiled Triton JIT Kern | v1.3.7 (r1) | d394b9df | 2026-10-16 |
| gdn_scan                               | triton_kernel | Recurrent Scan           | Standard        | 118.44    | 0.76              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| gla_scan                               | triton_kernel | Recurrent Scan           | Standard        | 114.30    | 0.67              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| sliding_window                         | triton_kernel | Sparse/Window Attn       | Standard        | 90.51     | 0.08              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| gqa                                    | triton_kernel | Attention Softmax        | Standard        | 23.38     | 0.21              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| fused_norm_linear                      | triton_kernel | Norm + GEMM / Gate       | Standard        | 22.52     | 0.02              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| flash_decoding                         | triton_kernel | Flash Decoding           | Standard        | 14.37     | 0.08              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| paged_attention                        | triton_kernel | Paged KV / State         | Standard        | 13.34     | 0.01              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| selective_scan                         | triton_kernel | Recurrent Scan           | Standard        | 10.89     | 5.02              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| fms_scan                               | triton_kernel | Recurrent Scan           | Standard        | 9.90      | 5.23              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| fused_kd                               | triton_kernel | Fused Loss               | Standard        | 8.70      | 0.56              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| hgrn2_scan                             | triton_kernel | Recurrent Scan           | Standard        | 7.44      | 0.02              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| cosformer_triton_kernel                | triton_kernel | Linear Attention         | Standard        | 4.52      | 0.05              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| _common / _types                       | triton_kernel | Pointers & Layout Safety | Standard        | 4.29      | 0.01              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| hardware_profile                       | triton_kernel | Hardware Telemetry       | Standard        | 4.22      | 0.01              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| op_registry                            | triton_kernel | Dispatch Subsystem       | Standard        | 4.17      | 0.01              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| autotuner                              | triton_kernel | Compiler Infrastructure  | Standard        | 3.75      | 0.01              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| block_sparse                           | triton_kernel | Sparse/Routing           | Standard        | 1.65      | 0.03              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| elementwise                            | triton_kernel | Elementwise/Activation   | Standard        | 1.49      | 0.20              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| softplus                               | triton_kernel | Elementwise/Activation   | Standard        | 1.23      | 0.04              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| fused_rope                             | triton_kernel | Positional Embedding     | Standard        | 1.00      | 0.08              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| varlen_attention                       | triton_kernel | Flash / Varlen Attn      | Standard        | 1.00      | 0.66              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| fused_ce                               | triton_kernel | Fused Loss               | Standard        | 0.98      | 2.35              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| state_update                           | triton_kernel | State Update             | Standard        | 0.60      | 0.06              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| sparse_attn                            | triton_kernel | Sparse/Routing           | Standard        | 0.26      | 0.40              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| low_rank                               | triton_kernel | Low-Rank SVD Fusion      | Standard        | 0.25      | 0.08              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
| linear_attn                            | triton_kernel | Linear Attention         | Standard        | 0.20      | 0.10              | -            | Triton vs PyTorch Benchm | v1.3.6 (r1) | 0911d2c0 | 2026-10-15 |
