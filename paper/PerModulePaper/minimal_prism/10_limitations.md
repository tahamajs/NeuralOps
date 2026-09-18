<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated. MinimalPRISM may behave differently on:
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

MinimalPRISM does not use calibration data. This means:
- No activation-aware weighting
- Rank is based on weight spectrum only

A3 and AASVD achieve 0.85% higher accuracy at the same compression, 
but require calibration. This is a deliberate trade-off.

## 10.5 No Explicit Reconstruction Bound per Layer

Unlike A3, MinimalPRISM does **not** enforce a Frobenius error threshold 
per layer. It relies on:
- Weight extraction checks (norm > 1e-4)
- Empirical accuracy after recovery
- Energy bound $\sqrt{1 - \eta}$ (theoretical, not enforced)

If a layer's energy-adaptive rank produces unexpectedly high error, this 
will only be visible in downstream accuracy. We recommend adding explicit 
Frobenius validation in future work.

## 10.6 Inference Latency on CPU

On CPU, factorized layers may be **slower** than dense due to:
- Two sequential matmuls
- Python overhead

On GPU, the situation is reversed (fewer FLOPs win).

## 10.7 Cross-Attention Not Supported

MinimalPRISM raises `NotImplementedError` for cross-attention. 
Encoder-decoder models (T5, BART) are not supported.

## 10.8 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% may 
be noise.

## 10.9 Rank Cap Heuristic

Default $r_\text{max} = 32$ is chosen empirically. There is no 
principled way to pick it without ablation.

## 10.10 Energy Threshold Heuristic

$\eta = 45.2$ is a heuristic. Optimal values may differ per model.

## 10.11 No Comparison with Some Baselines

We do not directly compare with:
- Performer / Linformer (attention replacement)
- SparseGPT / Wanda (pruning)
- QLoRA (quantization + LoRA)

## 10.12 Code Quality

Research-grade implementation. Known issues:
- Broad exception handling
- No unit tests for edge cases
- Numerical stability could be improved

## 10.13 Ethical Considerations

See Appendix E.

---