# NeuralOps Codebase Architecture: Part 02
# The 3-Level Kernel Dispatch Hierarchy & Zero Silent Fallback Law

**Assigned Subagent**: `Agent 02: Kernel Dispatch & Fallback Specialist`  
**Target Codebase Paths**:
- `neuralops/ops/cuda/_fallback.py` (Core 3-level fallback dispatcher & telemetry engine)
- `neuralops/core/cuda_kernels/_fallback.py` (Compatibility re-export shim)
- `neuralops/core/backend/dispatch.py` (High-level operator & device backend selection)
- `neuralops/core/hardware.py` (Memoized hardware topology probing & capability discovery)
- `neuralops/ops/triton/_common.py` & `neuralops/ops/triton/_types.py` (Triton dispatch & fallback reasons)
- `neuralops/ops/triton/op_registry.py` (TorchDynamo custom ops registration)
- `neuralops/ops/` (Tier 1 operator hierarchy: `cuda`, `triton`, `pytorch`)

---

## 1. Executive Summary & Design Philosophy

Deep learning operators face extreme execution diversity across production environments: from dedicated data centers with NVIDIA Hopper (SM 90) or Ada Lovelace (SM 8.9), to consumer workstations with partial driver setups, to CPU-only CI/CD runner nodes.

Part 02 details the **3-Level Kernel Dispatch Hierarchy** in NeuralOps:
- **Level 1 (Native CUDA C++ / HIP)**: Compiled C++/CUDA extensions in `neuralops.ops.cuda.*` achieving maximum throughput via hardware-level primitives (warp shuffles, tensor cores, shared memory double-buffering).
- **Level 2 (Triton JIT)**: Autotuned JIT kernels in `neuralops.ops.triton.*` compiled directly to PTX for FP16/BF16/FP32.
- **Level 3 (Vectorized PyTorch Reference)**: Universal pure-PyTorch mathematical references in `neuralops.ops.pytorch.*` and `*_pytorch_ref` ensuring 100% correct execution on any CPU or MPS environment.

Crucially, this system enforces the **Zero Silent Fallback Law**: a system failure in Level 1 or 2 must NEVER silently degrade to Level 3 without emitting structured warnings, failure reasons, and telemetry records.

---

## 2. Code Inventory & Actual Symbol Mapping

| File Path | Core Symbols / Functions | Architectural Responsibility |
|---|---|---|
| `neuralops/ops/cuda/_fallback.py` | `dispatch`, `force_kernel_backend`, `_DISPATCH_STATS`, `_LAST_DISPATCH_BACKEND`, `_LAST_DISPATCH_ERROR`, `_AutogradBridge` | Central low-level 3-level fallback dispatch engine with autograd bridging and telemetry |
| `neuralops/core/cuda_kernels/_fallback.py` | `dispatch`, `force_kernel_backend`, `get_dispatch_stats`, `reset_dispatch_stats` | Public compatibility facade re-exporting symbols from `neuralops.ops.cuda._fallback` |
| `neuralops/core/backend/dispatch.py` | `BackendDevice`, `available_backends`, `get_available_backend`, `get_backend`, `dispatch` | High-level operator routing (`chunk_gla`, `fused_linear_cross_entropy`, `fused_linear_kl_divergence`) |
| `neuralops/core/hardware.py` | `probe_hardware`, `HardwareTopology`, `HardwareBackend` | Memoized hardware topology and capability discovery with `@functools.lru_cache(maxsize=32)` |
| `neuralops/ops/triton/_common.py` | `triton_dispatch`, `pad_to_triton_power_of_2`, `_RUNTIME_FALLBACK_CACHE` | Triton operator router, head dimension padding, and runtime fallback circuit breaker |
| `neuralops/ops/triton/_types.py` | `check_triton_suitability`, `log_fallback_reason`, `FORCE_PYTORCH_FALLBACK` | Hardware capability validation, env flag checking, and warning deduplication |
| `neuralops/ops/triton/op_registry.py` | `_op_linear_attn`, `_op_sparse_attn`, `_op_low_rank`, `_op_gqa` | TorchDynamo / AOTAutograd custom ops registration preventing graph breaks |
| `neuralops/ops/cuda/__init__.py` | `list_cuda_kernels`, `CUDA_KERNEL_CATALOG`, PEP 562 `__getattr__` | Lazy-loading catalog of 110 CUDA kernel modules and 114 native implementations |

---

## 3. Mathematical & Algorithmic Contracts

### 3.1 3-Level Fallback State Machine

```
                        [User Operator Call]
                                 │
                                 ▼
                     [Fast Path: First Tensor CPU?]
                                 │
                   Yes ──────────┴────────── No
                    │                         │
                    ▼                         ▼
          [Level 3: PyTorch Ref]   [Forced Backend Active?]
                    │                         │
                    │           ┌─────────────┴─────────────┐
                    │      "cuda" / "triton"             "auto"
                    │           │                           │
                    │           ▼                           ▼
                    │    [Execute Forced Tier]     [Level 1: Native CUDA C++]
                    │     (Raise on Failure)         ├── Success ──► [Return Output]
                    │                                └── Failure:
                    │                                      ├── NEURALOPS_STRICT_CUDA=1 ──► [Raise]
                    │                                      └── Increment 'cuda_failures'
                    │                                            │
                    │                                            ▼
                    │                              [Level 2: Triton JIT]
                    │                                ├── Success ──► [Return Output]
                    │                                └── Failure:
                    │                                      └── Increment 'triton_failures'
                    │                                            │
                    │                                            ▼
                    └─────────────────────────────► [Level 3: PyTorch Reference]
                                                      ├── Success ──► [Return Output]
                                                      └── Failure ──► [Raise Exception]
```

### 3.2 Mathematical Tolerance Contract
For any tensor input $\mathbf{X}$ and parameter set $\boldsymbol{\Theta}$:
$$\max \left| \mathcal{O}_{\text{CUDA}}(\mathbf{X}) - \mathcal{O}_{\text{PyTorch}}(\mathbf{X}) \right| \le 10^{-3} + 10^{-3} \cdot |\mathcal{O}_{\text{PyTorch}}(\mathbf{X})|$$
$$\max \left| \mathcal{O}_{\text{Triton}}(\mathbf{X}) - \mathcal{O}_{\text{PyTorch}}(\mathbf{X}) \right| \le 10^{-3} + 10^{-3} \cdot |\mathcal{O}_{\text{PyTorch}}(\mathbf{X})|$$

Finite-difference autograd gradient checks must pass with $\text{atol} = 10^{-3}, \text{eps} = 10^{-5}$.

---

## 4. Implementation Analysis & Verification

### 4.1 Dispatch Routing (`neuralops/ops/cuda/_fallback.py`)
The canonical `dispatch()` function handles:
1. **CPU Fast-Path**: `_is_first_tensor_cpu(*args, **kwargs)` immediately executes `pytorch_fn` when tensors are on CPU, avoiding accelerator driver roundtrips and JIT overhead ($< 0.5\,\mu\text{s}$ dispatch latency).
2. **Backend Forcing**: Under `force_kernel_backend("cuda")` or `force_kernel_backend("triton")`, missing implementations or runtime errors raise `RuntimeError` immediately rather than silently falling through.
3. **Autograd Bridging (`_AutogradBridge`)**: When `has_grad = True`, wraps native C++ or Triton execution inside `torch.autograd.Function`, detaches forward tensors, and computes exact VJPs using the PyTorch reference during backward pass.
4. **Strict CUDA Control**: In forward evaluation, catches CUDA C++ failures, records `_LAST_DISPATCH_ERROR[kernel_name] = str(e)`, and re-raises immediately if `os.environ.get("NEURALOPS_STRICT_CUDA") == "1"`.

### 4.2 Telemetry Tracking & Counter Invariants
The global `_DISPATCH_STATS` dictionary records:
```python
_DISPATCH_STATS = {
    "cuda": 0,              # Successful Level 1 CUDA executions
    "triton": 0,            # Successful Level 2 Triton executions
    "pytorch": 0,           # Level 3 PyTorch fallback executions
    "cuda_failures": 0,     # Level 1 CUDA failure count
    "triton_failures": 0,   # Level 2 Triton failure count
}
```
Telemetry is accessed via `get_dispatch_stats()` and reset via `reset_dispatch_stats()`. Per-kernel backend resolution is tracked via `get_last_dispatch_backend(kernel_name)`.

### 4.3 Environment Flags
- `NEURALOPS_STRICT_CUDA=1`: Disables fallback on CUDA failure; raises immediately.
- `FORCE_PYTORCH_FALLBACK=1`: Globally forces Level 3 execution. Evaluated in `neuralops.ops.triton._types.check_triton_suitability` and `_fallback.FORCE_PYTORCH_FALLBACK`.
- `NEURALOPS_DEBUG_DISPATCH=1`: Streams real-time dispatch decisions and stack traces to `sys.stderr`.

### 4.4 Memoization and Latency Contracts
- **Hardware Topology**: `neuralops.core.hardware.probe_hardware` is decorated with `@functools.lru_cache(maxsize=32)`.
- **C++ Extension Loading**: Autograd modules cache JIT/compiled module singletons (`_MODULE = None`) so extension binding resolution occurs exactly once.
- **Triton Runtime Circuit Breaker**: `_RUNTIME_FALLBACK_CACHE: set[str]` in `neuralops/ops/triton/_common.py` caches failed kernels to bypass future Triton attempts.
- **Warning Deduplication**: `_WARNED_REASONS: set[str]` in `neuralops/ops/triton/_types.py` suppresses repeated duplicate log entries.

---

## 5. Audit Findings & Recommendations

1. **Logging Level Alignment**:
   - *Current State*: `neuralops/ops/cuda/_fallback.py` logs fallback failures using `logger.debug(...)` (and `sys.stderr` when `NEURALOPS_DEBUG_DISPATCH=1`).
   - *Standard Requirement*: AGENTS.md §7 requires that fallback from Level 1 or Level 2 emit a `logger.warning` with the failing level and reason to prevent silent performance degradation.
   - *Recommendation*: Elevate fallback log calls in `_fallback.py` to `logger.warning`.

2. **Autograd Path Strict CUDA Check**:
   - *Current State*: In `_fallback.py`, lines 242-252 catch CUDA exceptions in the autograd branch (`has_grad = True`) and fall back to Triton without checking `NEURALOPS_STRICT_CUDA`.
   - *Standard Requirement*: `NEURALOPS_STRICT_CUDA=1` should raise on Level 1 failure regardless of whether gradients are tracked.
   - *Recommendation*: Add `if os.environ.get("NEURALOPS_STRICT_CUDA") == "1": raise` to the autograd CUDA exception handler.

3. **Environment Variable Parsing in `_fallback.py`**:
   - *Current State*: `FORCE_PYTORCH_FALLBACK` in `_fallback.py` is a module boolean variable. It is checked as an environment variable in `triton/_types.py`, but not directly via `os.environ.get("FORCE_PYTORCH_FALLBACK")` in `_fallback.py:dispatch()`.
   - *Recommendation*: Harmonize by checking `FORCE_PYTORCH_FALLBACK or os.environ.get("FORCE_PYTORCH_FALLBACK", "0") in ("1", "true", "True")`.

---

## 6. Verification Evidence

- **Regression Tests**:
  - `tests/test_strict_backend_dispatch.py`: Verifies forced backend selection and auto fallback counter incrementation.
  - `tests/core/test_backend_forcing_regressions.py`: Verifies rejection of missing implementations, rejection of None returns, and unknown backend errors.
  - `tests/core/test_autograd_bridge_regressions.py`: Verifies analytical gradient propagation across multi-tensor outputs through `_AutogradBridge`.
  - `tests/test_cuda_triton_3level_complete.py`: 55 comprehensive operator tests on GPU across Level 1, Level 2, and Level 3.
- **Static Verification**:
  - All symbols confirmed present and importable in `neuralops.ops.cuda._fallback` and `neuralops.core.cuda_kernels._fallback`.
  - Dispatch stats invariant confirmed across all registered operations.
