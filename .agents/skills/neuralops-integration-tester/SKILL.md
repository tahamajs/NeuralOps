---
name: neuralops-integration-tester
description: "Verify multi-tier kernel dispatch fallback chains and end-to-end compression surgery pipelines."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "integration", "dispatch", "surgery", "pipeline"]
---

# Skill: Integration Tester

## When to use
Trigger when wiring new kernel dispatch tiers or validating end-to-end integration flows across surgery, training, and ledger logging.

## Prerequisites

## Steps
1. **Verify 3-Level Dispatch Fallback Hierarchy**:
   Validate graceful multi-tier resolution across Native CUDA C++ $\rightarrow$ Triton JIT $\rightarrow$ PyTorch Vectorized:
   ```python
   from neuralops.core.dispatch import force_kernel_backend

   with force_kernel_backend("cuda"):
       out_cuda = forward_op(x)
   with force_kernel_backend("triton"):
       out_triton = forward_op(x)
   with force_kernel_backend("pytorch"):
       out_ref = forward_op(x)
   ```
2. **Execute Full End-to-End Workflow**:
   Run full lifecycle pipeline:
   $\text{Compress} \longrightarrow \text{Surgery} \longrightarrow \text{Train / Calibrate} \longrightarrow \text{Distill Healing}$
3. **Validate Ledger Ingestion**:
   Confirm end-to-end execution generates valid record in `master_ledger.sqlite`.
4. **Verify Paper Table Automation**:
   Ensure `scripts/benchmarks/generate_all_paper_tables.py` incorporates the newly wired subsystem without error.

## Verification
- Confirm all fallback paths produce identical tensor shapes and cosine similarity $\ge 0.95$.

## Common pitfalls
- Silent fallback to CPU PyTorch without emitting backend warning logs.
- Breaking the dispatch signature across backend layers.
