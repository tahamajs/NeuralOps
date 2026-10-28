---
name: neuralops-perf-profiler
description: "Profile and optimize slow GPU kernels using Nsight Compute CLI metrics and SM tuning."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "ncu", "profiling", "cuda", "performance"]
---

# Skill: Performance Profiler

## When to use
Trigger when optimizing a slow or memory-bound CUDA C++ or Triton kernel to achieve speedup over baseline.

## Prerequisites
- NVIDIA Nsight Compute CLI (`ncu`) installed on GPU host

## Steps
1. **Profile Kernel via Nsight Compute**:
   Collect comprehensive hardware metric counters on remote GPU:
   ```bash
   ```
2. **Analyze Bottleneck Metrics**:
   - `achieved_occupancy`: Check active warps per SM vs theoretical maximum.
   - `dram_throughput`: Evaluate memory bandwidth utilization relative to hardware peak.
   - `branch_efficiency`: Identify divergent branch executions within warps.
3. **Apply Optimization Techniques**:
   - Shared memory (SMEM) tiling to reduce global memory transactions.
   - Warp-shuffle instructions (`__shfl_xor_sync`, `__shfl_down_sync`) for intra-warp reduction without shared memory round-trips.
   - TF32 / Tensor Core hardware instruction math mapping.
   - Adjust `__launch_bounds__` to maximize register allocation and occupancy.
4. **Verify Speedup Requirement**:
   Benchmark modified kernel to ensure `speedup >= 1.0x` over vectorized PyTorch baseline with latency `<= 20.0 ms`.

## Verification
- Re-run `ncu` to confirm improved DRAM throughput saturation and reduced stall cycles.

## Common pitfalls
- Over-allocating shared memory per thread block, depressing active occupancy.
- Ignoring register spilling to local memory.
