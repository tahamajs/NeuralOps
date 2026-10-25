---
name: neuralops-cuda-kernel-debugger
description: "Debug CUDA kernel numerical errors including NaN, Inf, and excessive differences."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "cuda", "debugger", "numerics"]
---

# Skill: CUDA Kernel Debugger

## When to use
Trigger when a CUDA kernel produces NaN, Inf, or numerical difference `max_diff > 1e-3` compared to reference.

## Prerequisites
- SSH access to `taha@192.168.1.100`
- `/mnt/d/venv/bin/python3`

## Steps
1. **Reproduce on GPU**:
   Run isolated numerical sanity check on remote GPU host:
   ```bash
   ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && /mnt/d/venv/bin/python3 tests/benchmarks/test_kernel_numerics.py --kernel <kernel_name>"
   ```
2. **Check 8 Common CUDA Bugs**:
   - FP16/BF16 intermediate accumulation overflow (ensure FP32 accumulation).
   - Division by zero or un-clamped normalization scale ($1 / \sqrt{d_k}$ with zero variance).
   - Out-of-bounds memory accesses in thread / block boundary tiles.
   - Uncoalesced global memory load/store strides across warps.
   - Shared memory bank conflicts or race conditions lacking `__syncthreads()`.
   - Missing or miscalculated warp shuffle reduction masks (`__activemask()` vs `0xffffffff`).
   - Pointer arithmetic overflow with dynamic offset indices.
   - Improper fast-math flags causing flush-to-zero / denormal divergence.
3. **Apply Fix & Verify Numerics**:
   Refactor kernel, compile, and verify `max_diff < 1e-3` (FP16) or `< 1e-4` (FP32).
4. **4-Hour Triage Limit**:
   If the kernel cannot be fixed and numerically stabilized within 4 hours, mark for deprecation and delete from mainline dispatch.

## Verification
- Re-run isolated test on remote GPU:
  `ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && /mnt/d/venv/bin/python3 -c 'import torch; from neuralops.core.cuda_kernels import <kernel>; ...'"`
- Confirm `max_diff < 1e-3` and zero NaNs/Infs.

## Common pitfalls
- Testing locally on macOS instead of remote RTX 4070 GPU host.
- Accumulating in FP16 inside tensor core warp routines instead of promoting to FP32.
- Forgetting to synchronize before reading from shared memory.
