# Kernels API Reference

NeuralOps implements **26 canonical Triton JIT kernel modules (housing 62 unified JIT operators across 4 architectural tiers)** and **112 Native CUDA C++ Packages (housing 114 native implementations)** delivering fused subquadratic sequence modeling, paged state management, and fused training objectives.

[![Release](https://img.shields.io/badge/release-v2.0.6-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v2.0.6)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![Interactive Roofline](https://img.shields.io/badge/%E2%9A%A1%20Interactive%20Roofline-RTX%204070-blue)](../roofline_interactive.html)
[![Hardware Verified](https://img.shields.io/badge/Hardware%20Verified-RTX%204070%20(SM%208.9)-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. 3-Level Fallback Hierarchy

Every compute operator in NeuralOps conforms to the strict 3-level fallback execution hierarchy:

```
┌────────────────────────────────────────────────────────┐
│ Level 1: Native CUDA C++ Extension (Max Throughput)    │
│  - NVCC compiled intrinsics, warp shuffles, tensor core│
│  - 112 packages housing 114 native implementations     │
└───────────────────────────┬────────────────────────────┘
                            │ (Fallback if CUDA uncompiled, with logger.warning)
┌───────────────────────────▼────────────────────────────┐
│ Level 2: Triton JIT Kernel (Autotuned GPU Compilation) │
│  - Autotuned 2D/3D tiles, FP16/BF16/FP32, zero block-ptr│
│  - 26 canonical modules housing 62 unified operators   │
└───────────────────────────┬────────────────────────────┘
                            │ (Fallback if non-CUDA or MPS/CPU, with logger.warning)
┌───────────────────────────▼────────────────────────────┐
│ Level 3: Vectorized PyTorch Reference (Universal CPU)  │
│  - Pure torch.Tensor fallback, exact numerical parity  │
└────────────────────────────────────────────────────────┘
```

Backend selection uses zero-allocation dispatch tables cached via `@lru_cache`. Backend forcing is accessible via `from neuralops import force_kernel_backend`.

---

## 2. Triton Kernel Catalog (26 Canonical Modules across 4 Tiers)

| Module File | Tier / Capability | Level 1 (CUDA) | Level 2 (Triton) | Level 3 (PyTorch) | Key Operators |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `gdn_scan.py` | Tier 1: Delta Scan | ✅ | ✅ | ✅ | `gdn_scan_triton`, `gdn_scan_pytorch` |
| `gla_scan.py` | Tier 1: Chunked GLA | ✅ | ✅ | ✅ | `chunk_gla_fwd`, `chunk_gla_bwd` |
| `hgrn2_scan.py` | Tier 1: Recurrent Scan | ✅ | ✅ | ✅ | `hgrn2_scan_triton`, `hgrn2_scan_pytorch` |
| `selective_scan.py` | Tier 1: State-Space SSM | ✅ | ✅ | ✅ | `selective_scan_triton`, `selective_scan_fn` |
| `fms_scan.py` | Tier 1: Factorised State | ✅ | ✅ | ✅ | `fms_recurrent_step_triton` |
| `linear_attn.py` | Tier 1: Linear Scan | ✅ | ✅ | ✅ | `fused_chunk_linear_scan` |
| `cosformer_triton_kernel.py` | Tier 1: Feature Projection | ✅ | ✅ | ✅ | `cosformer_triton` |
| `state_update.py` | Tier 1: Hidden Step | ✅ | ✅ | ✅ | `recurrent_state_update` |
| `paged_attention.py` | Tier 2: Paged KV/State | ✅ | ✅ | ✅ | `paged_attention_kernel` |
| `flash_decoding.py` | Tier 2: Parallel Decode | ✅ | ✅ | ✅ | `flash_decoding_stage1`, `flash_decoding_stage2` |
| `sliding_window.py` | Tier 2: Windowed Attention | ✅ | ✅ | ✅ | `sliding_window_triton` |
| `block_sparse.py` | Tier 2: Sparse Tiles | ✅ | ✅ | ✅ | `block_sparse_attention` |
| `sparse_attn.py` | Tier 2: Scatter Attention | ✅ | ✅ | ✅ | `topk_sparse_attention` |
| `gqa.py` | Tier 2: GQA Expansion | ✅ | ✅ | ✅ | `fused_gqa_triton` |
| `varlen_attention.py` | Tier 2: Ragged Batching | ✅ | ✅ | ✅ | `varlen_attention_triton` |
| `low_rank.py` | Tier 3: Low-Rank GEMM | ✅ | ✅ | ✅ | `fused_svd_gemm_triton`, `BitPackedW4A16Linear` |
| `elementwise.py` | Tier 3: FWHT & SwiGLU | ✅ | ✅ | ✅ | `fast_hadamard_transform`, `fused_swiglu` |
| `fused_norm_linear.py`| Tier 3: Fused Norm+GEMM | ✅ | ✅ | ✅ | `fused_norm_linear` |
| `fused_rope.py` | Tier 3: Rotary Embedding | ✅ | ✅ | ✅ | `fused_rope_embedding_kernel` |
| `softplus.py` | Tier 3: Stable Softplus | ✅ | ✅ | ✅ | `fused_softplus` |
| `fused_ce.py` | Tier 4: Cross-Entropy | ✅ | ✅ | ✅ | `fused_linear_cross_entropy` |
| `fused_kd.py` | Tier 4: Distillation KL | ✅ | ✅ | ✅ | `fused_distillation_loss` |
| `autotuner.py` | Core: Tile Autotuner | N/A | ✅ | ✅ | `neuralops_autotune` |
| `hardware_profile.py` | Core: Hardware Telemetry | N/A | ✅ | ✅ | `HardwareProfile`, `get_device_hardware_profile` |
| `op_registry.py` | Core: Op Registration | N/A | ✅ | ✅ | `register_kernel_op` |
| `_common.py` / `_types.py` | Core: Tensor Types | N/A | ✅ | ✅ | `should_use_triton`, `pad_to_triton_power_of_2` |

---

## 3. Native CUDA C++ Kernel Suite (141 Packages)

The native CUDA C++ kernel suite is implemented in `neuralops/core/cuda_kernels/` and `neuralops/ops/cuda/`. Key highlights:
- **141 Modular Kernel Packages**: Including `moe_routing`, `hgrn2_fused_bwd_scan`, `linear_attention`, `selective_scan`, `chunked_delta_gate_scan`, `paged_attention`, `sliding_window_attn`, `lut_codebook_dequant_gemm`, `flash_gqa`, `cosformer_linear_attention`, and `fused_rotary_emb`.
- **Warp-Level Shuffles**: `__shfl_xor_sync` and `__shfl_down_sync` eliminate shared memory roundtrips during reduction trees.
- **128-bit Vectorized Memory Loads**: `float4` and `int4` memory transactions maximizing DRAM bus utilization.
- **Shared Memory Padding**: Resolves bank conflicts across parallel warp access patterns.

---

## 4. Hardware Telemetry & Empirical Grounding

All kernels are empirically benchmarked on physical hardware:
- **Primary Physical Testbed**: **NVIDIA GeForce RTX 3090 (Ampere SM 8.6, GA102 silicon)**.
- **Physical Specifications**: 5,888 CUDA cores, 46 SMs, 184 Tensor Cores (4th gen), 160.2 TFLOPS boost / 141.6 base FP16, 504.0 GB/s GDDR6X, Arithmetic Intensity Ridge Point $I^* = 317.9\text{ FLOP/Byte}$, 12 GB VRAM.
- **Master Benchmark Dataset**: [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results).
