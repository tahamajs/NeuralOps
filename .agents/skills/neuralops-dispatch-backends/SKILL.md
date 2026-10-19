---
name: neuralops-dispatch-backends
description: Master the 3-level fallback hierarchy (CUDA C++/HIP -> Triton JIT -> Vectorized PyTorch), backend forcing with force_kernel_backend, and zero-overhead dispatch caching.
---

# NeuralOps Multi-Backend Dispatch and 3-Level Fallback Architecture

Use this skill when developing, refactoring, or diagnosing backend dispatch paths in NeuralOps. All high-performance operations in NeuralOps adhere to a strict **3-Level Fallback Hierarchy**, providing maximum hardware throughput on accelerators while guaranteeing 100% platform portability and graceful CPU execution.

---

## 1. The 3-Level Fallback Hierarchy

NeuralOps routes execution across three discrete implementation tiers:

```
+---------------------------------------------------------------------------------------------------+
|                                 NeuralOps 3-Level Execution Chain                                 |
+---------------------------------------------------------------------------------------------------+
|  [ LEVEL 1: NATIVE CUDA C++ / HIP ]                                                               |
|  * Source: `neuralops.core.cuda_kernels.*`                                                        |
|  * Target: NVIDIA Ada Lovelace / Hopper / Ampere (SM 8.0+) & AMD ROCm (gfx90a/942)                |
|  * Primitives: Warp shuffle intrinsics (__shfl_xor_sync), Tensor Core MMA, Shared Memory ping-pong |
|  * Strengths: Lowest kernel launch latency, hand-tuned register pressure, peak memory bandwidth   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         (On Verifier Rejection or Unavailable CUDA)
                                                  v
+---------------------------------------------------------------------------------------------------+
|  [ LEVEL 2: TRITON JIT KERNELS ]                                                                  |
|  * Source: `neuralops.core.triton_kernels.*`                                                      |
|  * Target: Accelerators supported by Triton compiler (SM 8.0+, AMD ROCm)                          |
|  * Primitives: Block-tiled pointers (`tl.block_type`), automated autotuning (@triton.autotune)    |
|  * Strengths: High productivity, dynamic block dimensions, automatic pipeline scheduling          |
+---------------------------------------------------------------------------------------------------+
                                                  |
                     (On Triton Compile Failure, Non-CUDA Device, or CPU)
                                                  v
+---------------------------------------------------------------------------------------------------+
|  [ LEVEL 3: VECTORIZED PYTORCH REFERENCE ]                                                        |
|  * Source: `neuralops.core.cuda_kernels._fallback.*_pytorch_ref`                                  |
|  * Target: Universal (CPU, Apple Silicon MPS, CUDA without compiler tools)                       |
|  * Primitives: Standard PyTorch vectorized tensor expressions, einops, torch.matmul              |
|  * Strengths: 100% bug-free mathematical oracle, CPU-safe, autograd ground truth                  |
+---------------------------------------------------------------------------------------------------+
```

### 1.1 Level 1: Native CUDA C++ / HIP Primitives
Native kernels are compiled into shared object libraries or JIT-loaded via `torch.utils.cpp_extension`. They leverage hardware-specific instructions:
```cpp
// Warp-level reduction intrinsic used in NeuralOps normalization & scan kernels
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### 1.2 Level 2: Triton JIT Kernel Implementation
Triton JIT kernels provide near-native throughput with high adaptability to ragged and block-sparse layouts:
```python
import triton
import triton.language as tl

@triton.jit
def _sliding_window_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_vb, stride_vh, stride_vm, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len, window_size,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Compute block row offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Bound sliding window span: [max(0, m - window_size), m]
    min_k = tl.maximum(0, pid_m * BLOCK_M - window_size)
    max_k = (pid_m + 1) * BLOCK_M

    # Initialize accumulation in FP32
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    # Block load, computation, and masked store
    # ...
```

### 1.3 Level 3: Vectorized PyTorch Reference
PyTorch references serve as the numerical oracle against which Level 1 and Level 2 are verified:
```python
def sliding_window_pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int) -> torch.Tensor:
    """Vectorized mathematical reference for sliding window attention."""
    B, H, T, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    # Construct band-diagonal causal sliding window mask
    row_idx = torch.arange(T, device=q.device).unsqueeze(1)
    col_idx = torch.arange(T, device=q.device).unsqueeze(0)
    dist = row_idx - col_idx
    mask = (dist >= 0) & (dist <= window_size)
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
```

---

## 2. Backend Forcing with `force_kernel_backend`

NeuralOps exposes a thread-safe context manager `force_kernel_backend` in `neuralops.core.cuda_kernels._fallback` to manually override runtime dispatch. This is essential for:
- Isolated benchmarking of individual backend tiers.
- Cross-backend numerical parity testing.
- Reproducing hardware-specific corner cases without changing application code.

### 2.1 The Backend Forcing API
```python
from neuralops.core.cuda_kernels import (
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
    get_last_dispatch_backend,
    get_last_dispatch_error,
)

# 1. Force execution through native CUDA C++
with force_kernel_backend("cuda"):
    out_cuda = sliding_window_attn(q, k, v, window_size=64)
assert get_last_dispatch_backend("sliding_window_attn") == "cuda"

# 2. Force execution through Triton JIT
with force_kernel_backend("triton"):
    out_triton = sliding_window_attn(q, k, v, window_size=64)
assert get_last_dispatch_backend("sliding_window_attn") == "triton"

# 3. Force execution through Vectorized PyTorch Reference
with force_kernel_backend("pytorch"):
    out_ref = sliding_window_attn(q, k, v, window_size=64)
assert get_last_dispatch_backend("sliding_window_attn") == "pytorch"
```

### 2.2 Telemetry and Dispatch Statistics
The runtime dispatcher tracks dispatch metrics across all invoked operators:
```python
stats = get_dispatch_stats()
# Returns:
# {
#     "cuda": 1420,
#     "triton": 350,
#     "pytorch": 12,
#     "cuda_failures": 0,
#     "triton_failures": 0
# }
```

### 2.3 Environment Variable Controls
Behavior can be governed at process startup:
- `NEURALOPS_STRICT_CUDA=1`: Disables silent fallback. If a CUDA C++ kernel raises an exception, the exception immediately bubbles up rather than falling back to Triton or PyTorch.
- `NEURALOPS_DEBUG_DISPATCH=1`: Logs real-time dispatch decisions and fallback reasons to `sys.stderr`.
- `FORCE_PYTORCH_FALLBACK=1`: Globally forces Level 3 execution for all kernels across the entire Python process.

---

## 3. Zero-Overhead Dispatch Caching & Autograd Bridging

Dispatch overhead can severely degrade throughput when processing small batch sizes or short sequence chunks. NeuralOps employs two architectural designs to eliminate this latency.

### 3.1 Fast-Path CPU Detection
CPU tensors bypass CUDA and Triton verifiers entirely:
```python
def dispatch(cuda_fn, triton_fn, pytorch_fn, *args, kernel_name="unknown", **kwargs):
    # Fast path: CPU input immediately takes Level 3 with zero driver or compilation overhead
    if _is_first_tensor_cpu(*args, **kwargs):
        _DISPATCH_STATS["pytorch"] += 1
        _LAST_DISPATCH_BACKEND[kernel_name] = "pytorch"
        return pytorch_fn(*args, **kwargs)
    # Accelerated dispatch continues...
```

### 3.2 TorchDynamo / torch.compile Compatibility
To prevent graph breaks under `torch.compile(model)`:
1. Dispatch verifiers must not perform data-dependent control flow or inspect tensor item values (`.item()`).
2. Custom autograd operations wrap CUDA / Triton executions inside standard `torch.autograd.Function` instances (`_AutogradBridge`), allowing TorchDynamo and AOTAutograd to treat them as opaque leaf nodes with registered VJPs.

### 3.3 The `_AutogradBridge` Mechanism
When a native Level 1 or Level 2 forward kernel is available, but backward gradients are required, `_AutogradBridge` seamlessly couples the high-speed forward kernel with automatic backward gradient computation:

```python
class _AutogradBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fwd_fn, bwd_ref_fn, reconstruct, *tensors):
        with torch.no_grad():
            args, kwargs = reconstruct(tensors)
            out = fwd_fn(*args, **kwargs)
        ctx.bwd_ref_fn = bwd_ref_fn
        ctx.reconstruct = reconstruct
        ctx.save_for_backward(*tensors)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        with torch.enable_grad():
            detached = [t.detach().requires_grad_(t.requires_grad) for t in saved]
            args, kwargs = ctx.reconstruct(detached)
            ref_out = ctx.bwd_ref_fn(*args, **kwargs)
            req_tensors = [t for t in detached if t.requires_grad]
            grads = torch.autograd.grad(ref_out, req_tensors, grad_output, allow_unused=True)

        grad_iter = iter(grads)
        tensor_grads = tuple(next(grad_iter) if t.requires_grad else None for t in detached)
        return (None, None, None, *tensor_grads)
```

---

## 4. Verification and Diagnostic CLI Workflow

Run these diagnostics to audit dispatch routing and verify 3-level fallback mechanics:

```bash
# 1. Run complete 3-level verification suite with debug output
NEURALOPS_DEBUG_DISPATCH=1 pytest tests/test_cuda_triton_3level_complete.py -v

# 2. Assert strict CUDA compliance across all operations
NEURALOPS_STRICT_CUDA=1 pytest tests/cuda_kernels/ -v

# 3. Test PyTorch fallback behavior under CPU simulation
FORCE_PYTORCH_FALLBACK=1 pytest tests/test_cuda_triton_3level_complete.py -v

# 4. Verify telemetry reset and counter tracking
python -c "from neuralops.core.cuda_kernels import get_dispatch_stats, reset_dispatch_stats; reset_dispatch_stats(); print(get_dispatch_stats())"
```
