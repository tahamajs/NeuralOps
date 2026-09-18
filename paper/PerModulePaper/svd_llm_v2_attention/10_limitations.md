<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated.

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |
| MoE | Not tested |

## 10.3 Hardware

Only tested on RTX 4070 Laptop (8 GB).

## 10.4 FP16 Numerical Sensitivity

Explicit storage of $\Sigma$ without distributing $\Sigma^{1/2}$ between 
the factors can be **less numerically stable in FP16** than symmetric 
storage. In our experiments, the difference is small (<0.3% accuracy), 
but for extreme FP16-only deployments, symmetric storage may be safer.

## 10.5 No Reconstruction Bound per Layer

SVD-LLM V2 does not enforce a Frobenius error threshold per layer. It relies 
on the energy bound $\sqrt{1 - \eta}$.

## 10.6 Calibration-Free Trade-off

No calibration data used. A3 and AASVD achieve 0.85% higher accuracy 
at the same compression but require calibration.

## 10.7 Inference Latency on CPU

On CPU, factorized layers may be slower than dense due to sequential matmuls.

## 10.8 Cross-Attention Not Supported

SVD-LLM V2 raises errors for cross-attention.

## 10.9 Single-Seed Results

All main results from a single seed. SE ~1.69%.

## 10.10 Heuristic Hyperparameters

$r_\text{max} = 32$ and $\eta = 45.2$ are empirical.

## 10.11 No Comparison with Some Baselines

We do not directly compare with:
- Performer / Linformer
- QLoRA

## 10.12 Code Quality

Research-grade. Known issues:
- Broad exception handling
- No unit tests for edge cases

## 10.13 Ethical Considerations

See Appendix E.

---