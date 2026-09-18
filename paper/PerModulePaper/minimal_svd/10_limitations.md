<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated. MinimalSVD may behave differently on:
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

Only tested on RTX 4070 Laptop (8 GB).

## 10.4 No Pre-Test Safety

MinimalSVD does **not** perform a pre-test on the first layer before mutating 
the model. If `copy_weights` fails on a middle layer, the model is left in a 
partially-injected state. Callers requiring pre-test safety should use 
MinimalPRISM instead.

## 10.5 No Whole-Model Traversal

MinimalSVD's injection hook targets a **single layer**. Whole-model 
traversal is the caller's responsibility. This is intentional (minimality) 
but requires more boilerplate in practice.

## 10.6 Calibration-Free Trade-off

MinimalSVD does not use calibration data. This means:
- No activation-aware weighting
- Rank is based on weight spectrum only

A3 and AASVD achieve 0.85% higher accuracy at the same compression, 
but require calibration.

## 10.7 No Explicit Reconstruction Bound per Layer

Unlike A3, MinimalSVD does **not** enforce a Frobenius error threshold per 
layer. It relies on:
- Weight extraction checks (norm > 1e-4)
- Empirical accuracy after recovery
- Energy bound $\sqrt{1 - \eta}$ (theoretical, not enforced)

## 10.8 Inference Latency on CPU

On CPU, factorized layers may be **slower** than dense due to two sequential 
matmuls.

## 10.9 Cross-Attention Not Supported

MinimalSVD raises errors for cross-attention. Encoder-decoder models (T5, 
BART) are not supported.

## 10.10 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% may 
be noise.

## 10.11 Heuristic Hyperparameters

Default $r_\text{max} = 32$ and $\eta = 45.2$ are chosen 
empirically. There is no principled way to pick them without ablation.

## 10.12 No Comparison with Some Baselines

We do not directly compare with:
- Performer / Linformer (attention replacement)
- SparseGPT / Wanda (pruning)
- QLoRA (quantization + LoRA)

## 10.13 Code Quality

Research-grade implementation. Known issues:
- Broad exception handling
- No unit tests for edge cases
- No double-injection guard

## 10.14 Ethical Considerations

See Appendix E.

---