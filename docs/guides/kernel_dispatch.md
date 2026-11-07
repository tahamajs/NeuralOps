# Kernel Dispatch & 3-Level Fallback Guide

NeuralOps enforces a strict, zero-overhead **3-Level Fallback Dispatch Hierarchy** for all performance-critical sequence modeling and matrix operations.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. The 3-Level Hierarchy Contract

Every kernel operation in NeuralOps adheres to a formal fallback contract:

```
┌────────────────────────────────────────────────────────┐
│ Level 1: Native CUDA C++ Extension (Max Throughput)    │
│  - NVCC compiled, 128-bit vectorized loads, warp shuff.│
│  - Target: NVIDIA SM >= 7.0 (Ampere, Ada Lovelace, etc)│
└───────────────────────────┬────────────────────────────┘
                            │ (Fallback if uncompiled or unsupported)
┌───────────────────────────▼────────────────────────────┐
│ Level 2: Triton JIT Kernel (Autotuned GPU Compilation) │
│  - Autotuned 2D/3D tiles, FP16/BF16/FP32, zero block-ptr│
│  - Target: Any CUDA / ROCm GPU with Triton support     │
└───────────────────────────┬────────────────────────────┘
                            │ (Fallback if non-CUDA or MPS/CPU)
┌───────────────────────────▼────────────────────────────┐
│ Level 3: Vectorized PyTorch Reference (Universal CPU)  │
│  - Exact mathematical parity, zero CUDA dependency     │
│  - Target: CPU, Apple Silicon (MPS), unit testing      │
└────────────────────────────────────────────────────────┘
```

### Fallback Logging Standards

In compliance with repo invariants:
- **No Silent Fallback**: If Level 1 fails, dispatch emits a `logger.warning` with the failing level and exact reason before attempting Level 2, and similarly from Level 2 to Level 3.
- **Zero-Allocation Cached Routing**: Backend resolution is cached via `@lru_cache` or zero-allocation static dispatch tables.

---

## 2. Kernel Suite Inventory

NeuralOps houses two complementary kernel suites:

1. **26 Canonical Triton Kernel Modules (58 files)**:
   - Reside in `neuralops/ops/triton/`.
   - Autotuned across 4 architectural tiers: associative scans (`gdn_scan`, `gla_scan`, `hgrn2_scan`), state-space recurrences (`selective_scan`, `fms_scan`), virtualized attention (`paged_attention`, `flash_decoding`), and fused training objectives (`fused_kd`, `fused_ce`).
2. **141 Native CUDA C++ Packages**:
   - Reside in `neuralops/core/cuda_kernels/` and `neuralops/ops/cuda/`.
   - Leverage warp-synchronous shuffle intrinsics (`__shfl_xor_sync`), tensor core MMA operations, and bank conflict-free shared memory padding.

---

## 3. Querying & Forcing Backends

### Inspecting Optimal Backend

```python
from neuralops.core.backend.dispatch import available_backends, get_backend

backends = available_backends()
backend = get_backend()
print(f"Available backends: {backends}, Selected: {backend}")  # 'cuda', 'triton', or 'cpu'
```

### Forcing a Specific Backend

For ablation studies, debugging, or CPU unit testing, use the context manager:

```python
import torch
from neuralops import force_kernel_backend
from neuralops.ops.triton.gdn_scan import gdn_scan_triton

# Force Triton execution
with force_kernel_backend("triton"):
    q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    beta = torch.sigmoid(torch.randn(2, 8, 512, device="cuda", dtype=torch.float16))
    alpha = torch.sigmoid(torch.randn(2, 8, 512, device="cuda", dtype=torch.float16))
    out = gdn_scan_triton(q, k, v, beta, alpha)

# Force pure PyTorch reference execution (works on CPU or Apple Silicon)
with force_kernel_backend("pytorch"):
    out_cpu = gdn_scan_triton(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), alpha.cpu())
```

---

## 4. Hardware Verification & Roofline Grounding

All kernels are verified for numerical stability and performance on:
- **Primary Testbed**: **NVIDIA GeForce RTX 4070 (SM 8.9 Ada Lovelace, AD104 silicon)**.
- **Hardware Telemetry**: 5,888 CUDA cores, 46 SMs, 184 Tensor Cores, 160.2 TFLOPS boost FP16, 504.0 GB/s GDDR6X, Ridge Point $I^* = 317.9\text{ FLOP/Byte}$, 12 GB VRAM.
- **Official Ledger**: [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results).
