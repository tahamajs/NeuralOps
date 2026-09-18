<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Fixed Rank

**Goal.** Use fixed rank $r = 64$ across all layers.

**Result.** Accuracy 84.8 vs energy-adaptive 88.88.

**Conclusion.** Energy-adaptive is better.

### D.1.2 Full-Rank Factorization

**Goal.** Use $r = \min(d_\text{in}, d_\text{out})$ to avoid any loss.

**Result.** No parameter reduction (ratio = 1.0×).

**Conclusion.** Useless.

### D.1.3 Cross-Attention Support

**Goal.** Extend to encoder-decoder models.

**Result.** Complex; deferred.

**Conclusion.** Not implemented.

### D.1.4 No LoRA Recovery

**Goal.** Skip LoRA, rely on SVD alone.

**Result.** Accuracy dropped to 86.4\%%.

**Conclusion.** LoRA recovery essential.

### D.1.5 Adaptive Rank via Retry

**Goal.** Increase rank if error exceeds threshold (like A3).

**Result.** Slower; ~same accuracy as energy-adaptive.

**Conclusion.** Energy is simpler.

## D.2 Abandoned Variants

### D.2.1 Activation-Aware Rank

We tested selecting rank based on activation energy (like AASVD). Result: 
requires calibration, no improvement.

### D.2.2 Per-Head Rank

We tested different rank per attention head. Result: no improvement, 
complexity increase.

### D.2.3 Low-Rank Bias

We tested factorizing bias as well. Result: negligible parameter savings.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 (unstable) |

### Energy

| $\eta$ | DistilBERT | Note |
|---|---|---|
| 0.90 | 0.85 | Too aggressive |
| 0.95 | 0.85 | Sweet spot |
| 0.99 | 0.85 | Marginal gain |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Orthogonal init for U/V | Slow convergence | No change |
| LayerNorm recalibration | Distribution shift | <0.3% gain |
| Weight tying between Q/K | Parameter sharing | Accuracy drop |
| Adaptive β in LoRA | Training stability | No change |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Companion Variant Comparison

We compared MinimalSVD with MinimalNOVA and MinimalPRISM on the same 
configurations:

| Variant | DistilBERT | RoBERTa |
|---|---|---|
| MinimalNOVA | 0.85% | 32% |
| MinimalPRISM | 0.85% | 32% |
| MinimalSVD | 88.88% | 79.36% |

**Observation.** Accuracy differences are within seed noise (±1%). The core 
algorithm is identical across all three variants.

## D.7 Lessons Learned

1. **Energy-adaptive rank is sufficient.**
2. **LoRA recovery is essential.**
3. **Fixed rank is competitive but less flexible.**
4. **Cross-attention is complex; not needed for main results.**
5. **No calibration required — a feature, not a bug.**
6. **Injection protocol differences (SVD vs NOVA vs PRISM) do not affect 
   accuracy** — only engineering robustness.

---