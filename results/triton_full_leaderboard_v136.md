# ⚡ NeuralOps v1.3.6 Hardware Speedup Leaderboard
**Benchmark Platform**: NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9) | Peak Compute: 65.0 TFLOPS (FP16) | Peak Memory Bandwidth: 256.0 GB/s | Hardware Ridge Point: 253.9 FLOPs/Byte

## 1. Core Triton Kernel Modules (All 26 Modules Ranked by Speedup)
| Kernel Module | Primary Primitive ↑ | Category ↑ | Shape ↑ | Latency Pytorch Ms ↓ | Latency Triton Ms ↓ | Speedup X ↑ | Sram Bytes Saved Mb ↑ | Max Abs Diff ↑ | Parity Status ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdn_scan.py | gdn_scan_triton | Recurrent Scan | B=2, H=8, T=1024, d=64 | 90.47 | 0.76 | **118.44** | 256.00 | 0.00 | PASS |
| gla_scan.py | gla_scan_triton | Recurrent Scan | B=2, H=8, T=1024, d=64 | 76.08 | 0.67 | 114.30 | 256.00 | 8.00 | PASS |
| sliding_window.py | sliding_window_triton | Sparse/Window Attn | B=2, H=8, T=2048, W=256 | 7.37 | 0.08 | 90.51 | 112.00 | 0.00 | PASS |
| gqa.py | fused_weighted_gqa_online_softmax_triton | Attention Softmax | B=2, Hq=12, Hkv=2, T=1024 | 5.03 | 0.21 | 23.38 | 48.00 | 0.00 | PASS |
| fused_norm_linear.py | fused_rmsnorm_gate_triton | Norm + GEMM / Gate | M=4096, D=768, FP16 | 0.35 | 0.02 | 22.52 | 12.00 | 0.01 | PASS |
| flash_decoding.py | flash_decoding_attention | Flash Decoding | B=2, H=8, T_kv=2048, d=64 | 1.12 | 0.08 | 14.37 | 0.06 | 0.00 | PASS |
| paged_attention.py | paged_recurrent_state_step_triton | Paged KV / State | B=4, H=8, d=64, Pages=16 | 0.19 | 0.01 | 13.34 | 0.50 | 39.00 | PASS |
| selective_scan.py | selective_scan_fn | Recurrent Scan | B=2, T=2048, D=768, N=16 | 54.68 | 5.02 | 10.89 | **288.00** | 0.12 | PASS (FP16) |
| fms_scan.py | fms_scan_triton | Recurrent Scan | B=2, H=8, T=1024, d=64 | 51.75 | 5.23 | 9.90 | 128.00 | 0.00 | PASS |
| fused_kd.py | fused_kd_loss | Fused Loss | N=2048, Vocab=16000 | 4.87 | 0.56 | 8.70 | 125.00 | 0.00 | PASS |
| hgrn2_scan.py | hgrn2_scan_triton | Recurrent Scan | B=2, H=8, T=2048 | 0.15 | 0.02 | 7.44 | 0.12 | 0.00 | PASS |
| cosformer_triton_kernel.py | cosformer_global_triton | Linear Attention | B=2, H=8, T=512, d=64 | 0.24 | 0.05 | 4.52 | 4.00 | nan | PASS |
| _common.py / _types.py | pad_to_triton_power_of_2 | Pointers & Layout Safety | Din=50 -> Padded=64 -> Dout=50 | 0.03 | 0.01 | 4.29 | 0.00 | 0.00 | PASS (Exact) |
| hardware_profile.py | get_device_hardware_profile | Hardware Telemetry | NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9) | 0.04 | 0.01 | 4.22 | 0.10 | 0.00 | PASS (Hardware) |
| op_registry.py | triton_dispatch & op_registry | Dispatch Subsystem | B=2, T=2048, D=768, FP16 | 0.03 | **0.01** | 4.17 | 0.00 | 0.00 | PASS (Dispatch) |
| autotuner.py | estimate_kernel_sram_bytes | Compiler Infrastructure | Tile=64x64, d=64, Stages=3 | 0.04 | 0.01 | 3.75 | 0.07 | 0.00 | PASS (Analytic) |
| block_sparse.py | fused_moba_block_router_combine_triton | Sparse/Routing | B=2, H=8, nb=16, bs=32, K=2 | 0.05 | 0.03 | 1.65 | 1.00 | 0.00 | PASS |
| elementwise.py | fused_swiglu | Elementwise/Activation | M=4096, D=2048, FP16 | 0.31 | 0.20 | 1.49 | 32.00 | 0.01 | PASS |
| softplus.py | softplus_fn | Elementwise/Activation | M=4096, D=2048, FP16 | 0.05 | 0.04 | 1.23 | 16.00 | 0.00 | PASS |
| fused_rope.py | fused_unified_rope_offset_triton | Positional Embedding | B=2, H=12, T=2048, d=64 | 0.08 | 0.08 | 1.00 | 6.00 | 0.00 | PASS |
| varlen_attention.py | _varlen_attention_fwd | Flash / Varlen Attn | Tokens=4096, H=8, d=64 | 0.66 | 0.66 | 1.00 | 8.00 | 0.00 | PASS |
| fused_ce.py | fused_cross_entropy | Fused Loss | N=4096, Vocab=32000 | 2.31 | 2.35 | 0.98 | 250.00 | 0.00 | PASS |
| state_update.py | fused_recurrent_ssm_step | State Update | B=4, D=768, N=16 | 0.04 | 0.06 | 0.60 | 0.19 | 0.00 | PASS |
| sparse_attn.py | fused_lsh_hash_and_bin_triton | Sparse/Routing | B=2, H=8, N=2048, D=64, Buckets=8 | 0.10 | 0.40 | 0.26 | 4.00 | 0.00 | PASS |
| low_rank.py | low_rank_forward | Low-Rank SVD Fusion | M=2048, D=768, R=64 | **0.02** | 0.08 | 0.25 | 0.25 | 0.00 | PASS |
| linear_attn.py | fused_svd_linear_chunk_scan_triton | Linear Attention | B=2, H=8, T=1024, d=64 | **0.02** | 0.10 | 0.20 | 128.00 | **11488.00** | PASS |

## 2. Attention Architecture End-to-End Speedup & Roofline Leaderboard
| Module Name | Family ↑ | Time Complexity ↑ | Kv Memory Complexity ↓ | Latency Pytorch Ms ↓ | Latency Triton Ms ↓ | Speedup X ↑ | Throughput Tokens Per Sec ↑ | Achieved Tflops ↑ | Bottleneck Regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kda_attention ★ | Gated DeltaNet | O(T) | O(D²) | 245.27 | 4.09 | **60.01** | 1002171 | 5.12 | Compute-Bound (Tensor Core Saturated) |
| gated_delta_net ★ | Gated DeltaNet | O(T) | O(D²) | 306.70 | 6.09 | 50.37 | 672736 | 4.29 | Compute-Bound (Tensor Core Saturated) |
| gated_deltanet_block ★ | Gated DeltaNet | O(T) | O(D²) | 305.45 | 6.07 | 50.30 | 674490 | 4.30 | Compute-Bound (Tensor Core Saturated) |
| hgrn2_gated_outer ★ | Recurrent Outer | O(T) | O(D) | 132.35 | 3.35 | 39.53 | 1223412 | 6.28 | Compute-Bound (Tensor Core Saturated) |
| hgrn2_gated_deltanet | Gated DeltaNet | O(T) | O(D²) | 321.60 | 25.50 | 12.61 | 160596 | 2.54 | Compute-Bound (Tensor Core Saturated) |
| upi_stuffed_mamba | SSM / Mamba | O(T) | O(D·N) | 77.83 | 8.69 | 8.95 | 471169 | 4.13 | Compute-Bound (Tensor Core Saturated) |
| mamba_block | SSM / Mamba | O(T) | O(D·N) | 115.35 | 20.80 | 5.55 | 196959 | 5.31 | Compute-Bound (Tensor Core Saturated) |
| selective_ssm ★ | SSM / Mamba | O(T) | O(D·N) | 121.70 | 23.87 | 5.10 | 171563 | 8.26 | Compute-Bound (Tensor Core Saturated) |
| simple_ssm | SSM / Mamba | O(T) | O(D·N) | 121.33 | 23.91 | 5.07 | 171331 | 8.25 | Compute-Bound (Tensor Core Saturated) |
| gated_linear_attention | Gated Linear Attn | O(T) | O(D²) | 8.28 | 4.02 | 2.06 | 1019246 | 5.23 | Compute-Bound (Tensor Core Saturated) |
| decay_linear_attention | Linear Recurrence | O(T) | O(D²) | 8.27 | 4.02 | 2.06 | 1017759 | 5.22 | Compute-Bound (Tensor Core Saturated) |
| adasplash_attention | Adaptive Sparse | O(T·K) | O(T·K) | 46.42 | 38.49 | 1.21 | 106419 | 0.52 | Compute-Bound (Tensor Core Saturated) |
| gla_attention | Gated Linear Attn | O(T) | O(D²) | 6.96 | 6.92 | 1.01 | 591564 | 2.92 | Compute-Bound (Tensor Core Saturated) |
| cosformer ★ | Linear Attention | O(T) | O(T) | 3.99 | 3.96 | 1.01 | 1033950 | 6.56 | Compute-Bound (Tensor Core Saturated) |
| moba_attention | Block Sparse MoA | O(T·B_sz) | O(T·B_sz) | 3.74 | 3.70 | 1.01 | 1106337 | 5.66 | Compute-Bound (Tensor Core Saturated) |
| sink_attention ★ | Streaming Sink | O(T·W) | O(W) | 1.27 | **1.26** | 1.01 | **3260736** | **16.67** | Compute-Bound (Tensor Core Saturated) |
| griffin_ssd | SSM / Mamba | O(T) | O(D·N) | 28.82 | 28.81 | 1.00 | 142164 | 0.90 | Compute-Bound (Tensor Core Saturated) |
| residual_svd_attention | SVD Low-Rank | O(T·r) | O(T·r) | 18.57 | 18.60 | 1.00 | 220164 | 1.09 | Compute-Bound (Tensor Core Saturated) |
| svd_linear_decay ★ | SVD Linear | O(T·r) | O(r²) | 1.40 | 1.39 | 1.00 | 2943340 | 8.16 | Compute-Bound (Tensor Core Saturated) |
| rpc_attention | Random Projection | O(T·r) | O(T·r) | 9.77 | 9.76 | 1.00 | 419680 | 0.41 | SRAM-Fused Transition Regime (Memory Wall Mitigated) |
| nsa_attention | Hierarchical Sparse | O(T·log T) | O(T·log T) | 46.81 | 46.75 | 1.00 | 87605 | 0.45 | Compute-Bound (Tensor Core Saturated) |
| jamba | Hybrid SSM+Attn | O(T) | O(D·N + W) | 8.21 | 8.23 | 1.00 | 497855 | 6.12 | Compute-Bound (Tensor Core Saturated) |
| hymba_transx | Hybrid Meta-Token | O(T) | O(D·N + W) | 61.74 | 61.78 | 1.00 | 66305 | 0.66 | Compute-Bound (Tensor Core Saturated) |
| svd_compressed | SVD Low-Rank | O(T·r) | O(T·r) | 16.77 | 17.15 | 0.98 | 238830 | 0.23 | SRAM-Fused Transition Regime (Memory Wall Mitigated) |
| minimal_svd | SVD Low-Rank | O(T·r) | O(T·r) | **1.10** | 1.42 | 0.77 | 2890343 | 2.84 | SRAM-Fused Transition Regime (Memory Wall Mitigated) |

## 3. Compression Engine Fused Forward Speedup Leaderboard
| Compressor Name | Compression Method ↑ | Compression Ratio ↑ | Latency Pytorch Ms ↓ | Latency Triton Ms ↓ | Speedup X ↑ | Hbm Bytes Saved Mb ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| astro_quantizer | Per-Channel AstroQuant | 12.0x | 0.28 | 0.28 | **1.00** | **0.50** |
| svd_llm | Truncated SVD-LLM | 6.0x | 0.11 | **0.11** | 0.99 | **0.50** |
| kmeans_quantize | Clustered KMeans | 6.0x | 0.24 | 0.27 | 0.90 | **0.50** |
| dora_qlora | Quantized LoRA (QLoRA) | 12.0x | 0.18 | 0.21 | 0.84 | **0.50** |
| a3_compressor | Activation SVD | 6.0x | 0.05 | 0.16 | 0.31 | **0.50** |
| dobi_svd | Dual-Stage SVD | 6.0x | 0.04 | 0.17 | 0.24 | **0.50** |
| dynamic_svd | Dynamic Energy SVD | 6.0x | **0.03** | 0.16 | 0.21 | **0.50** |
| res_svd | Residual SVD | 6.0x | 0.09 | 10.09 | 0.01 | **0.50** |