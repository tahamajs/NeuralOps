---
name: neuralops-benchmark-auditor
description: "Audit and classify 108 CUDA kernels and 26 Triton kernels against strict performance gates."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "benchmark", "auditor", "triage"]
---

# Skill: Benchmark Auditor

## When to use
Trigger when executing repository-wide audits across 108 CUDA C++ kernels or 26 Triton kernel modules.

## Prerequisites
- SSH access to `taha@192.168.1.100`
- `/mnt/d/venv/bin/python3`

## Steps
1. **Run Isolated Benchmark Audit**:
   Execute the strict classifier on remote GPU:
   ```bash
   ssh taha@192.168.1.100 "cd /mnt/d/SVD_Project && /mnt/d/venv/bin/python3 scripts/benchmarks/benchmark_all_108_cuda_isolated.py --classify-strict"
   ```
2. **Classify Kernel Outputs**:
   Categorize every operator into one of 6 strict states:
   - `PASS`: Meets all 5 quality criteria (`ref_ms >= 0.01`, `speedup >= 1.0x`, `max_diff < 0.01`, `cuda_ms <= 20.0`, no NaNs).
   - `FAIL_NAN`: Produces NaN or Inf values in forward/backward pass.
   - `FAIL_NUMERICS`: Absolute difference `max_diff >= 0.01` (FP16) or `>= 1e-4` (FP32).
   - `NO_BASELINE`: Missing or unvectorized PyTorch reference (`ref_ms < 0.01`).
   - `FAIL_SLOW`: Speedup `< 1.0x` compared to PyTorch baseline.
   - `FAIL_LATENCY`: Forward latency `cuda_ms > 20.0`.
3. **Triage Action Plan**:
   - `DELETE`: Unfixable or architectural dead-ends.
   - `DEBUG_4H`: Allocate 4 hours to resolve numerical or launch bounds issues.
   - `FIX_BASELINE`: Rewrite unvectorized or loop-based PyTorch baseline.
4. **Report Counts**:
   Aggregate and print breakdown summary table across all categories.

## Verification
- Inspect generated report and ensure 100% of target kernels are audited without silent failures.

## Common pitfalls
- Running audit on CPU or non-SM 8.9 GPU.
- Masking slow kernels by artificially inflating reference runtime.
