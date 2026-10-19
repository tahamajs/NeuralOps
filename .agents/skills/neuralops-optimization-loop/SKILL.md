---
name: neuralops-optimization-loop
description: Execute rigorous, reproducible NeuralOps kernel optimization loops with frozen correctness gates (torch.autograd.gradcheck), cold-cache microbenchmarking, and structured candidate comparison logging.
---

# NeuralOps Optimization Loop

Use this skill for disciplined, hypothesis-driven kernel optimization across NeuralOps operators. The optimization loop guarantees that every proposed speedup is accompanied by mathematical correctness proof, statistically robust microbenchmark evidence, and structured telemetry.

Follow [neuralops-nvidia-performance](../neuralops-nvidia-performance/SKILL.md) or [neuralops-ascend-performance](../neuralops-ascend-performance/SKILL.md) for platform-specific profiler recipes.

---

## 1. The Optimization Loop Protocol

Every performance iteration follows a strict five-stage gate:

```mermaid
flowchart TD
    Freeze["1. Freeze Baseline & Gates<br/>(Record SHA, lock clocks, freeze test matrices)"]
    Hypothesis["2. Profile & Formulate Single Hypothesis<br/>(NCU / msprof bottleneck identification)"]
    Gate["3. Run Frozen Correctness Gate<br/>(torch.autograd.gradcheck & NaN allocation poison)"]
    Bench["4. Reproducible Microbenchmarking<br/>(L2 cache flush, CUDA events, statistical IQR)"]
    Log["5. Structured Candidate Logging<br/>(Candidate JSON log, keep/drop decision)"]
    
    Freeze --> Hypothesis --> Gate
    Gate -->|Pass| Bench
    Gate -->|Fail: Numerical Divergence| Drop["Drop Candidate & Log Failure"]
    Bench -->|Statistically Significant Gain (> 3-sigma)| Promote["Promote Candidate & Update Baseline"]
    Bench -->|Insignificant Gain or Regression| Drop
    Promote --> Hypothesis
    Drop --> Hypothesis
```

---

## 2. Frozen Correctness Gates and Autograd Validation

Never evaluate benchmark timing on unverified code. Correctness gates must remain frozen throughout an optimization sprint:
- **No relaxing tolerances**: Do not widen `atol` or `rtol` to accommodate a candidate.
- **No skipping parameter sweeps**: Test matrices must cover dense, ragged/varlen, small boundary lengths ($L=1, 15, 31$), and large production shapes ($L=8192, 16384$).
- **NaN Memory Poisoning**: Run under NeuralOps test suites where uninitialized allocations are poisoned with NaNs.

### Complete `torch.autograd.gradcheck` Test Harness

Custom backward kernels (e.g. `chunk_kda_bwd`) must pass analytical vs. numerical finite-difference verification in `float64`:

```python
# Copyright (c) 2024-2026, Taha Majlesi
# NeuralOps Systems Engineering Group
"""
Rigorous Analytical Autograd Gradient Check Harness for NeuralOps Kernels.
Verifies dQ, dK, dV, dAlpha, dBeta against finite-difference Jacobians.
"""
from __future__ import annotations
import torch
import torch.testing

def run_kda_gradcheck(
    op_callable,
    batch_size: int = 2,
    num_heads: int = 2,
    seq_len: int = 32,
    dim_k: int = 16,
    dim_v: int = 16,
    device: str = "cuda",
) -> None:
    """
    Executes torch.autograd.gradcheck in float64 precision.
    Tolerances are frozen at: eps=1e-6, atol=1e-4, rtol=1e-3.
    """
    torch.manual_seed(42)

    # Initialize double precision tensors for stable finite differences
    q = torch.randn(batch_size, num_heads, seq_len, dim_k, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, dim_k, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len, dim_v, device=device, dtype=torch.float64, requires_grad=True)
    
    # Raw gates bounded to prevent numerical overflow during finite perturbation
    alpha = torch.empty(batch_size, num_heads, seq_len, device=device, dtype=torch.float64).uniform_(-3.0, -0.5).requires_grad_(True)
    beta = torch.empty(batch_size, num_heads, seq_len, device=device, dtype=torch.float64).uniform_(0.1, 0.9).requires_grad_(True)

    print("Running torch.autograd.gradcheck (analytical vs numerical Jacobian)...")
    
    passed = torch.autograd.gradcheck(
        func=op_callable,
        inputs=(q, k, v, alpha, beta),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
        check_undefined_grad=True,
        check_grad_dtypes=True,
    )
    print("✓ Gradcheck PASSED successfully: All analytical gradients verified.")

def verify_numerical_tolerances(
    candidate_out: torch.Tensor,
    reference_out: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """
    Strict numerical validation matching NeuralOps testing standards.
    """
    thresholds = {
        torch.float32: {"atol": 1e-5, "rtol": 1e-4},
        torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
        torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    }
    t = thresholds.get(dtype, {"atol": 1e-3, "rtol": 1e-3})
    
    torch.testing.assert_close(
        candidate_out,
        reference_out,
        atol=t["atol"],
        rtol=t["rtol"],
        check_dtype=True,
        equal_nan=False,
    )
    print(f"✓ Numerical match verified against reference oracle for {dtype}.")
```

### SVD State Compression Accuracy Verification

When optimizing state compression (e.g. projecting states into rank $R$), track relative Frobenius error:

$$\text{Relative Reconstruction Error} = \frac{\| S_t - U_t \Sigma_t V_t^\top \|_F}{\| S_t \|_F} \le \tau_{\text{SVD}}$$

```python
def verify_svd_state_approximation(
    s_exact: torch.Tensor,
    u_mat: torch.Tensor,
    s_core: torch.Tensor,
    v_mat: torch.Tensor,
    max_relative_error: float = 0.05,  # 5% Frobenius error ceiling
) -> None:
    """Verifies that low-rank SVD compression preserves recurrent memory dynamics."""
    # Reconstruct state: U @ S_core @ V^T
    s_approx = torch.matmul(torch.matmul(u_mat, s_core), v_mat.transpose(-1, -2))
    
    frob_diff = torch.linalg.norm(s_exact.float() - s_approx.float(), ord="fro")
    frob_norm = torch.linalg.norm(s_exact.float(), ord="fro")
    rel_error = (frob_diff / torch.clamp(frob_norm, min=1e-6)).item()

    assert rel_error <= max_relative_error, (
        f"SVD state compression exceeded tolerance: "
        f"relative error {rel_error:.4f} > allowed {max_relative_error:.4f}"
    )
    print(f"✓ SVD low-rank compression verified: relative Frobenius error = {rel_error:.2%}")
```

---

## 3. Reproducible Microbenchmarking Protocols

Measuring latency without controlling environmental noise produces false positives. The NeuralOps microbenchmark harness enforces:

1. **GPU Clock Locking**: Freeze core and memory clocks to prevent thermal throttling and dynamic frequency scaling:
   ```bash
   # NVIDIA: Lock clocks (e.g. RTX 4090 at 2520 MHz core, 10501 MHz mem)
   sudo nvidia-smi -pm 1
   sudo nvidia-smi -lgc 2520,2520
   sudo nvidia-smi -lmc 10501,10501
   ```
2. **L2 Cache Flushing**: GPUs cache matrix blocks in large L2 caches (e.g. 72 MB on RTX 4090). Benchmarking without cache flushing measures unrealistic warm-cache performance. Allocate a separate dummy buffer ($2\times$ L2 size, e.g. 256 MB) and overwrite it before every timed invocation.
3. **CUDA Event Timing**: Use native CUDA events rather than Python `time.perf_counter()` to isolate host scheduling jitter.
4. **Statistical Trimming**: Execute $\ge 25$ warmup iterations and $\ge 100$ timed runs. Compute the 20% trimmed mean, median, and Interquartile Range (IQR).

### Robust Microbenchmarking Script with L2 Flushing

```python
# Copyright (c) 2024-2026, Taha Majlesi
# NeuralOps Systems Engineering Group
"""
Robust, Cold-Cache CUDA Microbenchmark Runner for NeuralOps Operators.
"""
from __future__ import annotations
import numpy as np
import torch

class ColdCacheBenchmarkRunner:
    def __init__(self, l2_cache_size_mb: int = 72):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Allocate 2x L2 cache size to ensure complete cache displacement
        dummy_elements = (l2_cache_size_mb * 1024 * 1024 * 2) // 4
        self.dummy_l2_flush = torch.empty(dummy_elements, device=self.device, dtype=torch.float32)

    def flush_l2_cache(self) -> None:
        """Overwrites dummy buffer to completely evict L2 cache lines."""
        self.dummy_l2_flush.zero_()

    def benchmark_operator(
        self,
        op_fn,
        warmup_runs: int = 25,
        timed_runs: int = 100,
        flush_cache: bool = True,
    ) -> dict[str, float]:
        # 1. Warmup phase (compiles Triton kernels, warms memory controllers)
        for _ in range(warmup_runs):
            op_fn()
        torch.cuda.synchronize()

        latencies_ms = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 2. Synchronized timing loop
        for _ in range(timed_runs):
            if flush_cache:
                self.flush_l2_cache()
                torch.cuda.synchronize()

            start_event.record()
            op_fn()
            end_event.record()
            torch.cuda.synchronize()

            latencies_ms.append(start_event.elapsed_time(end_event))

        data = np.array(latencies_ms)
        q25, median, q75 = np.percentile(data, [25, 50, 75])
        mean = np.mean(data)
        std_dev = np.std(data)

        # 20% Trimmed Mean (removes outliers)
        lower_cut = int(0.10 * len(data))
        upper_cut = int(0.90 * len(data))
        trimmed_mean = np.mean(np.sort(data)[lower_cut:upper_cut])

        return {
            "median_ms": float(median),
            "trimmed_mean_ms": float(trimmed_mean),
            "mean_ms": float(mean),
            "std_ms": float(std_dev),
            "iqr_ms": float(q75 - q25),
            "min_ms": float(np.min(data)),
            "max_ms": float(np.max(data)),
        }
```

---

## 4. Candidate Comparison and Telemetry Logging

Track optimization history in a structured format outside source code comments. Maintain a persistent JSON log in `profile/<op>_opt/candidate_history.json`.

### Candidate Telemetry Schema

```json
{
  "candidate_id": "cand_kda_sm89_v3",
  "base_sha": "a1b2c3d4e5f67890",
  "timestamp": "2026-10-19T10:00:00Z",
  "operator": "chunk_kda_fwd",
  "hardware": {
    "device": "NVIDIA GeForce RTX 4090",
    "compute_capability": "8.9",
    "clocks_locked_mhz": 2520
  },
  "workload": {
    "batch_size": 2,
    "num_heads": 8,
    "seq_len": 8192,
    "dim_k": 64,
    "dim_v": 64,
    "dtype": "bfloat16"
  },
  "hypothesis": "Apply 8-element padding to shared memory Q/K tiles to eliminate bank conflicts.",
  "correctness": {
    "gradcheck_passed": true,
    "assert_close_passed": true,
    "nan_poison_tested": true
  },
  "profiler_metrics": {
    "registers_per_thread": 118,
    "local_memory_spill_bytes": 0,
    "shared_memory_bank_conflicts": 0,
    "tensor_core_utilization_pct": 54.2
  },
  "benchmark_results": {
    "baseline_median_ms": 1.482,
    "candidate_median_ms": 1.194,
    "speedup_ratio": 1.241,
    "statistically_significant": true,
    "std_dev_sigma": 4.8
  },
  "decision": "PROMOTED"
}
```

### Promotion Decision Rule

A candidate is marked **`PROMOTED`** only if:
1. All frozen correctness gates (`gradcheck`, `assert_close`, NaN poisoning) are $100\%$ passed.
2. Latency improvement exceeds $3\sigma$ of measurement noise:
   $$\text{Baseline}_{\text{median}} - \text{Candidate}_{\text{median}} > 3 \times \sqrt{\sigma_{\text{base}}^2 + \sigma_{\text{cand}}^2}$$
3. Zero local memory register spills (`local_memory_spill_bytes == 0`).
4. Backward compatibility across supported sequence lengths and head dimension configurations is preserved.
