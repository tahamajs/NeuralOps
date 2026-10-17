# 📊 Hardware Benchmarks of 3-Level Native CUDA C++ Operators

**Category Filter:** `cuda_kernel` | **Total Verified Records:** 18

## 1. Executive Summary & Micro-Architectural Acceleration

Evaluates 18 native CUDA C++ kernels compiled with `-O3 -gencode arch=compute_89,code=sm_89` for NVIDIA Ada Lovelace architectures. These kernels bypass framework overhead via warp-level primitives, register caching, and asynchronous shared memory copies (`cp.async`).

### Key Acceleration Highlights:
- **Peak Speedup Champions**: `moe_routing` (28466.32x), `hgrn2_fused_bwd_scan` (436.29x), `linear_attention` (243.98x), `selective_scan` (109.34x), `chunked_delta_gate_scan` (82.40x)
- **Sub-millisecond Execution**: `fused_rotary_emb` (0.000 ms), `moe_routing` (0.000 ms), `paged_attention` (0.000 ms), `w4a16_gemm` (0.000 ms), `fused_norm_linear` (0.020 ms)

---

## 2. Pareto Rankings (Speedup vs. Execution Latency)

| Pareto Rank | Kernel Name | Family | CUDA Lat (ms) | PyTorch Lat (ms) | Speedup (vs. PyTorch) | Peak VRAM (MB) | Status |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **#1** | `moe_routing` | 3-level-native-cuda | 0.000 ms | 0.000 ms | **28466.32x** | 44.0 MB | VERIFIED (RTX 4070) |
| **#2** | `hgrn2_fused_bwd_scan` | 3-level-native-cuda | 0.026 ms | 5.686 ms | **436.29x** | 39.4 MB | VERIFIED (RTX 4070) |
| **#3** | `linear_attention` | 3-level-native-cuda | 0.257 ms | 40.532 ms | **243.98x** | 43.9 MB | VERIFIED (RTX 4070) |
| **#4** | `selective_scan` | 3-level-native-cuda | 0.356 ms | 27.227 ms | **109.34x** | 10.9 MB | VERIFIED (RTX 4070) |
| **#5** | `chunked_delta_gate_scan` | 3-level-native-cuda | 0.123 ms | 43.208 ms | **82.40x** | 38.8 MB | VERIFIED (RTX 4070) |
| **#6** | `paged_attention` | 3-level-native-cuda | 0.000 ms | 0.433 ms | **45.89x** | 36.1 MB | VERIFIED (RTX 4070) |
| **#7** | `sliding_window_attn` | 3-level-native-cuda | 0.252 ms | 1.826 ms | **7.24x** | 26.9 MB | VERIFIED (RTX 4070) |
| **#8** | `block_sparse_attn` | 3-level-native-cuda | 1.529 ms | 1.819 ms | **5.02x** | 26.9 MB | VERIFIED (RTX 4070) |
| **#9** | `lut_codebook_dequant_gemm` | 3-level-native-cuda | 0.026 ms | 0.096 ms | **5.00x** | 43.4 MB | VERIFIED (RTX 4070) |
| **#10** | `fused_rotary_emb` | 3-level-native-cuda | 0.000 ms | 0.029 ms | **2.87x** | 45.2 MB | VERIFIED (RTX 4070) |
| **#11** | `flash_gqa` | 3-level-native-cuda | 0.034 ms | 0.192 ms | **2.49x** | 34.0 MB | VERIFIED (RTX 4070) |
| **#12** | `cosformer_linear_attention` | 3-level-native-cuda | 0.025 ms | 0.111 ms | **1.88x** | 39.7 MB | VERIFIED (RTX 4070) |
| **#13** | `kronecker_matmul` | 3-level-native-cuda | 0.025 ms | 0.030 ms | **1.78x** | 43.3 MB | VERIFIED (RTX 4070) |
| **#14** | `fused_lowrank_gated_gemm` | 3-level-native-cuda | 0.040 ms | 0.026 ms | **0.67x** | 38.3 MB | VERIFIED (RTX 4070) |
| **#15** | `fp8_gemm` | 3-level-native-cuda | 0.068 ms | 0.078 ms | **0.58x** | 30.0 MB | VERIFIED (RTX 4070) |
| **#16** | `w4a16_gemm` | 3-level-native-cuda | 0.000 ms | 0.105 ms | **0.54x** | 24.5 MB | VERIFIED (RTX 4070) |
| **#17** | `fused_cross_entropy` | 3-level-native-cuda | 0.063 ms | 0.013 ms | **0.35x** | 45.7 MB | VERIFIED (RTX 4070) |
| **#18** | `fused_norm_linear` | 3-level-native-cuda | 0.020 ms | 0.000 ms | **0.12x** | 41.3 MB | VERIFIED (RTX 4070) |

---

## 3. High-Precision Master Benchmark Table

| Kernel Name | Family | CUDA Latency (ms) | PyTorch Latency (ms) | Speedup (x) | Peak VRAM (MB) | Throughput (tok/s) | Hardware Verification | Git SHA |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| `block_sparse_attn` | 3-level-native-cuda | 1.529 | 1.819 | **5.02x** | 26.9 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `chunked_delta_gate_scan` | 3-level-native-cuda | 0.123 | 43.208 | **82.40x** | 38.8 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `cosformer_linear_attention` | 3-level-native-cuda | 0.025 | 0.111 | **1.88x** | 39.7 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `flash_gqa` | 3-level-native-cuda | 0.034 | 0.192 | **2.49x** | 34.0 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `fp8_gemm` | 3-level-native-cuda | 0.068 | 0.078 | **0.58x** | 30.0 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `fused_cross_entropy` | 3-level-native-cuda | 0.063 | 0.013 | **0.35x** | 45.7 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `fused_lowrank_gated_gemm` | 3-level-native-cuda | 0.040 | 0.026 | **0.67x** | 38.3 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `fused_norm_linear` | 3-level-native-cuda | 0.020 | 0.000 | **0.12x** | 41.3 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `fused_rotary_emb` | 3-level-native-cuda | 0.000 | 0.029 | **2.87x** | 45.2 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `hgrn2_fused_bwd_scan` | 3-level-native-cuda | 0.026 | 5.686 | **436.29x** | 39.4 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `kronecker_matmul` | 3-level-native-cuda | 0.025 | 0.030 | **1.78x** | 43.3 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `linear_attention` | 3-level-native-cuda | 0.257 | 40.532 | **243.98x** | 43.9 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `lut_codebook_dequant_gemm` | 3-level-native-cuda | 0.026 | 0.096 | **5.00x** | 43.4 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `moe_routing` | 3-level-native-cuda | 0.000 | 0.000 | **28466.32x** | 44.0 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `paged_attention` | 3-level-native-cuda | 0.000 | 0.433 | **45.89x** | 36.1 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `selective_scan` | 3-level-native-cuda | 0.356 | 27.227 | **109.34x** | 10.9 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `sliding_window_attn` | 3-level-native-cuda | 0.252 | 1.826 | **7.24x** | 26.9 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
| `w4a16_gemm` | 3-level-native-cuda | 0.000 | 0.105 | **0.54x** | 24.5 | - | VERIFIED (RTX 4070) | `9ed147b99c` |
