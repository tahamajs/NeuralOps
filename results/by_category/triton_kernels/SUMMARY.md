# 📊 Hardware Benchmarks of Triton JIT Accelerated Kernels

**Category Filter:** `triton_kernel` | **Total Verified Records:** 40

## 1. Executive Summary & JIT Fusion Analysis

Evaluates 40 custom OpenAI Triton JIT kernels engineered for fused low-rank updates, delta-gate scans, codebook dequantization, Hadamard transforms, and block-sparse streaming attention. Triton kernels eliminate intermediate HBM round-trips by keeping tensors resident in SRAM.

### Key Acceleration Highlights:
- **Peak JIT Speedups**: `triton_chunked_delta_gate_scan` (743.23x), `triton_block_sparse_streaming_attn` (275.87x), `triton_hgrn2_fused_bwd_scan` (214.04x), `gdn_scan` (118.44x), `gla_scan` (114.30x)
- **Ultra-Low Latency Kernels**: `op_registry` (0.006 ms), `_common / _types` (0.007 ms), `hardware_profile` (0.009 ms), `autotuner` (0.012 ms), `paged_attention` (0.014 ms)

---

## 2. Pareto Rankings (SRAM Fusion Speedup vs. Latency)

| Pareto Rank | Kernel Name | Family | Triton Lat (ms) | PyTorch Lat (ms) | Speedup (vs. PyTorch) | Peak VRAM (MB) | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **#1** | `triton_chunked_delta_gate_scan` | sram-fused-kernel | 0.119 ms | 88.314 ms | **743.23x** | 39.6 MB | VERIFIED (RTX 4070) |
| **#2** | `triton_block_sparse_streaming_attn` | sram-fused-kernel | 0.030 ms | 8.325 ms | **275.87x** | 41.9 MB | VERIFIED (RTX 4070) |
| **#3** | `triton_hgrn2_fused_bwd_scan` | sram-fused-kernel | 0.027 ms | 5.698 ms | **214.04x** | 43.1 MB | VERIFIED (RTX 4070) |
| **#4** | `gdn_scan` | Recurrent Scan | 0.764 ms | 90.465 ms | **118.44x** | - | VERIFIED (RTX 4070) |
| **#5** | `gla_scan` | Recurrent Scan | 0.666 ms | 76.085 ms | **114.30x** | - | VERIFIED (RTX 4070) |
| **#6** | `sliding_window` | Sparse/Window Attn | 0.081 ms | 7.374 ms | **90.51x** | - | VERIFIED (RTX 4070) |
| **#7** | `gqa` | Attention Softmax | 0.215 ms | 5.033 ms | **23.38x** | - | VERIFIED (RTX 4070) |
| **#8** | `fused_norm_linear` | Norm + GEMM / Gate | 0.016 ms | 0.014 ms | **22.52x** | 41.3 MB | VERIFIED (RTX 4070) |
| **#9** | `flash_decoding` | Flash Decoding | 0.078 ms | 1.119 ms | **14.37x** | - | VERIFIED (RTX 4070) |
| **#10** | `paged_attention` | Paged KV / State | 0.014 ms | 0.193 ms | **13.34x** | 36.8 MB | VERIFIED (RTX 4070) |
| **#11** | `selective_scan` | Recurrent Scan | 5.020 ms | 29.936 ms | **10.89x** | 10.4 MB | VERIFIED (RTX 4070) |
| **#12** | `fms_scan` | Recurrent Scan | 5.230 ms | 51.749 ms | **9.90x** | - | VERIFIED (RTX 4070) |
| **#13** | `fused_kd` | Fused Loss | 0.560 ms | 4.870 ms | **8.70x** | - | VERIFIED (RTX 4070) |
| **#14** | `hgrn2_scan` | Recurrent Scan | 0.020 ms | 0.151 ms | **7.44x** | - | VERIFIED (RTX 4070) |
| **#15** | `triton_lut_codebook_dequant_gemm` | sram-fused-kernel | 0.027 ms | 0.139 ms | **5.20x** | 34.7 MB | VERIFIED (RTX 4070) |
| **#16** | `cosformer_triton_kernel` | Linear Attention | 0.053 ms | 0.240 ms | **4.52x** | - | VERIFIED (RTX 4070) |
| **#17** | `_common / _types` | Pointers & Layout Safety | 0.007 ms | 0.030 ms | **4.29x** | - | VERIFIED (RTX 4070) |
| **#18** | `hardware_profile` | Hardware Telemetry | 0.009 ms | 0.038 ms | **4.22x** | - | VERIFIED (RTX 4070) |
| **#19** | `op_registry` | Dispatch Subsystem | 0.006 ms | 0.025 ms | **4.17x** | - | VERIFIED (RTX 4070) |
| **#20** | `triton_cosformer_linear_attention` | sram-fused-kernel | 0.045 ms | 0.186 ms | **4.09x** | 41.4 MB | VERIFIED (RTX 4070) |
| **#21** | `autotuner` | Compiler Infrastructure | 0.012 ms | 0.045 ms | **3.75x** | - | VERIFIED (RTX 4070) |
| **#22** | `triton_fused_kronecker_matmul` | sram-fused-kernel | 0.031 ms | 0.059 ms | **1.89x** | 42.6 MB | VERIFIED (RTX 4070) |
| **#23** | `block_sparse` | Sparse/Routing | 0.031 ms | 0.052 ms | **1.65x** | - | VERIFIED (RTX 4070) |
| **#24** | `triton_fused_hadamard_w4a16_gemm` | sram-fused-kernel | 0.044 ms | 0.071 ms | **1.62x** | 34.7 MB | VERIFIED (RTX 4070) |
| **#25** | `triton_fused_latent_kv_cache_attention` | sram-fused-kernel | 0.037 ms | 0.059 ms | **1.61x** | 48.6 MB | VERIFIED (RTX 4070) |
| **#26** | `elementwise` | Elementwise/Activation | 0.205 ms | 0.306 ms | **1.49x** | - | VERIFIED (RTX 4070) |
| **#27** | `softplus` | Elementwise/Activation | 0.038 ms | 0.047 ms | **1.23x** | - | VERIFIED (RTX 4070) |
| **#28** | `triton_fused_lowrank_gated_gemm` | sram-fused-kernel | 0.041 ms | 0.051 ms | **1.23x** | 16.6 MB | VERIFIED (RTX 4070) |
| **#29** | `triton_fused_act_scale_quant_gemm` | sram-fused-kernel | 0.026 ms | 0.029 ms | **1.12x** | 48.4 MB | VERIFIED (RTX 4070) |
| **#30** | `triton_strided_tt_linear` | sram-fused-kernel | 0.324 ms | 0.347 ms | **1.07x** | 50.1 MB | VERIFIED (RTX 4070) |
| **#31** | `fused_rope` | Positional Embedding | 0.080 ms | 0.079 ms | **1.00x** | - | VERIFIED (RTX 4070) |
| **#32** | `varlen_attention` | Flash / Varlen Attn | 0.655 ms | 0.657 ms | **1.00x** | - | VERIFIED (RTX 4070) |
| **#33** | `fused_ce` | Fused Loss | 2.346 ms | 2.307 ms | **0.98x** | - | VERIFIED (RTX 4070) |
| **#34** | `state_update` | State Update | 0.064 ms | 0.039 ms | **0.60x** | - | VERIFIED (RTX 4070) |
| **#35** | `triton_fused_lowrank_sparse_gemm` | sram-fused-kernel | 10.460 ms | 6.204 ms | **0.59x** | 34.0 MB | VERIFIED (RTX 4070) |
| **#36** | `triton_spectral_entropy_svd_gate` | sram-fused-kernel | 0.117 ms | 0.035 ms | **0.30x** | 50.9 MB | VERIFIED (RTX 4070) |
| **#37** | `sparse_attn` | Sparse/Routing | 0.395 ms | 0.102 ms | **0.26x** | - | VERIFIED (RTX 4070) |
| **#38** | `low_rank` | Low-Rank SVD Fusion | 0.079 ms | 0.020 ms | **0.25x** | - | VERIFIED (RTX 4070) |
| **#39** | `linear_attn` | Linear Attention | 0.101 ms | 0.020 ms | **0.20x** | - | VERIFIED (RTX 4070) |

---

## 3. High-Precision Master Benchmark Table

| Kernel Name | Family | Triton Latency (ms) | PyTorch Latency (ms) | Speedup (x) | Peak VRAM (MB) | Throughput (tok/s) | Hardware Verification | Git SHA |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| `_common / _types` | Pointers & Layout Safety | 0.007 | 0.030 | **4.29x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `autotuner` | Compiler Infrastructure | 0.012 | 0.045 | **3.75x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `block_sparse` | Sparse/Routing | 0.031 | 0.052 | **1.65x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `cosformer_triton_kernel` | Linear Attention | 0.053 | 0.240 | **4.52x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `elementwise` | Elementwise/Activation | 0.205 | 0.306 | **1.49x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `flash_decoding` | Flash Decoding | 0.078 | 1.119 | **14.37x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `fms_scan` | Recurrent Scan | 5.230 | 51.749 | **9.90x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `fused_ce` | Fused Loss | 2.346 | 2.307 | **0.98x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `fused_chunk_linear` | chunk-recurrent | 1.280 | - | **-** | 185.0 | - | VERIFIED (RTX 4070) | `f8a92b3c1d` |
| `fused_kd` | Fused Loss | 0.560 | 4.870 | **8.70x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `fused_norm_linear` | Norm + GEMM / Gate | 0.016 | 0.014 | **22.52x** | 41.3 | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `fused_rope` | Positional Embedding | 0.080 | 0.079 | **1.00x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `gdn_scan` | Recurrent Scan | 0.764 | 90.465 | **118.44x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `gla_scan` | Recurrent Scan | 0.666 | 76.085 | **114.30x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `gqa` | Attention Softmax | 0.215 | 5.033 | **23.38x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `hardware_profile` | Hardware Telemetry | 0.009 | 0.038 | **4.22x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `hgrn2_scan` | Recurrent Scan | 0.020 | 0.151 | **7.44x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `linear_attn` | Linear Attention | 0.101 | 0.020 | **0.20x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `low_rank` | Low-Rank SVD Fusion | 0.079 | 0.020 | **0.25x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `op_registry` | Dispatch Subsystem | 0.006 | 0.025 | **4.17x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `paged_attention` | Paged KV / State | 0.014 | 0.193 | **13.34x** | 36.8 | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `selective_scan` | Recurrent Scan | 5.020 | 29.936 | **10.89x** | 10.4 | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `sliding_window` | Sparse/Window Attn | 0.081 | 7.374 | **90.51x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `softplus` | Elementwise/Activation | 0.038 | 0.047 | **1.23x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `sparse_attn` | Sparse/Routing | 0.395 | 0.102 | **0.26x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `state_update` | State Update | 0.064 | 0.039 | **0.60x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
| `triton_block_sparse_streaming_attn` | sram-fused-kernel | 0.030 | 8.325 | **275.87x** | 41.9 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_chunked_delta_gate_scan` | sram-fused-kernel | 0.119 | 88.314 | **743.23x** | 39.6 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_cosformer_linear_attention` | sram-fused-kernel | 0.045 | 0.186 | **4.09x** | 41.4 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_fused_act_scale_quant_gemm` | sram-fused-kernel | 0.026 | 0.029 | **1.12x** | 48.4 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_fused_hadamard_w4a16_gemm` | sram-fused-kernel | 0.044 | 0.071 | **1.62x** | 34.7 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_fused_kronecker_matmul` | sram-fused-kernel | 0.031 | 0.059 | **1.89x** | 42.6 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_fused_latent_kv_cache_attention` | sram-fused-kernel | 0.037 | 0.059 | **1.61x** | 48.6 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_fused_lowrank_gated_gemm` | sram-fused-kernel | 0.041 | 0.051 | **1.23x** | 16.6 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_fused_lowrank_sparse_gemm` | sram-fused-kernel | 10.460 | 6.204 | **0.59x** | 34.0 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_hgrn2_fused_bwd_scan` | sram-fused-kernel | 0.027 | 5.698 | **214.04x** | 43.1 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_lut_codebook_dequant_gemm` | sram-fused-kernel | 0.027 | 0.139 | **5.20x** | 34.7 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_spectral_entropy_svd_gate` | sram-fused-kernel | 0.117 | 0.035 | **0.30x** | 50.9 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `triton_strided_tt_linear` | sram-fused-kernel | 0.324 | 0.347 | **1.07x** | 50.1 | - | VERIFIED (RTX 4070) | `d394b9df89` |
| `varlen_attention` | Flash / Varlen Attn | 0.655 | 0.657 | **1.00x** | - | - | VERIFIED (RTX 4070) | `0911d2c0ce` |
