# Known Sub-Optimal Kernels & Recommended Backends

In accordance with NeuralOps Core Operating Principles (§0 and §7), NeuralOps preserves custom CUDA and Triton implementations demonstrating theoretical algorithmic formulations even when hardware-specific cuBLAS / ATen / cuDNN libraries currently execute faster on specific tensor shapes.

When `backend='auto'` (default), NeuralOps emits a performance advisory warning and routes execution to the optimal high-throughput engine. To explicitly force execution of the custom research kernel for experimental profiling or comparative benchmarking, callers may use `with force_kernel_backend('cuda'):`.

## Sub-Optimal Kernels Summary

| # | Kernel Name | Subsystem / File | Benchmarked Relative Speedup | Root Cause of Slowdown | Recommended Backend (`auto` default) | When to Use Custom Kernel |
|---|---|---|:---:|---|---|---|
| 1 | `fused_norm_linear` | `neuralops/ops/cuda/fused_norm_linear` | **0.004x** (244x slower) | Serial thread loop over $K$, uncoalesced global strided memory loads, and redundant normalization across $(M, N)$ output coordinates without 2D SRAM tiling. | `pytorch` (cuBLAS GEMM) | Algorithm research on fused reduction or educational inspection of combined LayerNorm + GEMM. |
| 2 | `fp8_gemm` | `neuralops/ops/cuda/fp8_gemm` | **0.15x** (6.7x slower) | Software bitwise register decoding of FP8 values (`decode_4_e4m3`) into FP32 registers incurs substantial ALU instruction pressure compared to native FP8 Tensor Core MMA instructions. | `pytorch` (`torch._scaled_mm` on SM 8.9+, cuBLAS FP16 reference elsewhere) | Direct inspection of software FP8 decompression pipelines on GPUs lacking hardware FP8 instructions (SM < 8.9). |
| 3 | `fused_cross_entropy` | `neuralops/ops/cuda/fused_ce` | **0.31x** (3.2x slower) | Loss reductions across vocabulary dimension ($V \ge 32,000$) are strictly memory bandwidth bound. PyTorch `F.cross_entropy` utilizes cuDNN/ATen optimized kernels saturating 90% DRAM roofline. | `pytorch` (`F.cross_entropy`) | End-to-end memory profiling or fused custom label smoothing experiments. |
| 4 | `w4a16_gemm` | `neuralops/ops/cuda/quantized_gemm` | **0.39x** (2.5x slower) | Dynamic inner-loop bit-shift unpacking (`>> 4`, `& 0x0F`) increases register pressure and drops GPU warp occupancy below 25%. | `pytorch` (Vectorized dequantization or Marlin / AWQ) | Validating INT4 uniform quantization math without external library dependencies. |
| 5 | `speculative_verify` | `neuralops/ops/cuda/speculative_verify` | **0.53x** (1.9x slower) | Warp thread divergence across variable draft candidate matching loops serializes execution on warp lanes. | `pytorch` (Vectorized tensor equality `cumprod`) | Low-level token verification benchmarking and SIMD warp divergence analysis. |
| 6 | `fused_lowrank_gated_gemm` | `neuralops/ops/cuda/low_rank_svd` | **0.65x** (1.5x slower) | Allocation of intermediate bottleneck matrix $Z$ in device global memory rather than executing single-pass shared memory tile fusion. | `pytorch` (Two sequential cuBLAS `torch.matmul` calls) | Hardware memory footprint profiling and non-linear gating research. |
