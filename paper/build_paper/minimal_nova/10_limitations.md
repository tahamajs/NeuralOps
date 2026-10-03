<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated. MinimalNOVA may behave differently on:
- Generation tasks
- Multi-class classification
- Token-level tasks
- Long-context tasks

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |
| MoE | Not tested |

## 10.3 Hardware

Only tested on RTX 4070 Laptop (8 GB). Scaling to other GPUs is unknown.

## 10.4 Calibration-Free Trade-off

MinimalNOVA does not use calibration data. This means:
- No activation-aware weighting
- Rank is based on weight spectrum only

**A3 and AASVD** achieve 0.85% higher accuracy at the same 
compression, but require calibration. This is a deliberate trade-off.

## 10.5 No Reconstruction Validation

Unlike A3, MinimalNOVA does **not** enforce a Frobenius error threshold 
per layer. It relies on:
- Weight extraction checks (norm > 1e-4)
- Empirical accuracy after recovery

If a layer's energy-adaptive rank produces unexpectedly high error, this 
will only be visible in downstream accuracy. We recommend adding explicit 
Frobenius validation in future work.

## 10.6 Inference Latency on CPU

On CPU, factorized layers may be **slower** than dense due to:
- Two sequential matmuls
- Python overhead

On GPU, the situation is reversed (fewer FLOPs win).

## 10.7 Cross-Attention Not Supported

MinimalNOVA raises `NotImplementedError` for cross-attention. Encoder-decoder 
models (T5, BART) are not supported.

## 10.8 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% may 
be noise.

## 10.9 No Formal Downstream Accuracy Bound

We provide a **reconstruction** error bound (Proposition 4.10.2), but no 
bound on downstream task accuracy. The gap between reconstruction error and 
task accuracy is not characterized.

## 10.10 Rank Cap Heuristic

The default $r_\text{max} = 32$ is chosen empirically. There is no 
principled way to pick it without ablation.

## 10.11 Energy Threshold Choice

$\eta = 45.2$ is a heuristic. Optimal values may differ per model.

## 10.12 No Comparison with Some Baselines

We do not directly compare with:
- Performer / Linformer (attention replacement)
- SparseGPT / Wanda (pruning)
- QLoRA (quantization + LoRA)

## 10.13 Code Quality

Research-grade implementation. Known issues:
- Broad exception handling in `copy_weights`
- No unit tests for edge cases
- Numerical stability could be improved

## 10.14 Ethical Considerations

See Appendix E.

---