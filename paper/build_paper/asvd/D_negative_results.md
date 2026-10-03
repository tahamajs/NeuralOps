<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Full Covariance Weighting

**Goal.** Replace diagonal weighting with full covariance (AASVD-style).

**Result.** 5× slower, <0.3% improvement.

**Conclusion.** Diagonal is sufficient for ASVD.

### D.1.2 Very Low Rank (r=8)

**Goal.** Maximize compression.

**Result.** Accuracy dropped by 8.0%. Non-recoverable.

**Conclusion.** r=16 is the practical minimum.

### D.1.3 Tight Error Threshold (τ=0.05)

**Goal.** Strict reconstruction bound.

**Result.** 60% of layers aborted. Injection failed.

**Conclusion.** τ=0.20 is more practical.

### D.1.4 No LoRA Recovery

**Goal.** See if compression alone is sufficient.

**Result.** Accuracy dropped by 8.0%.

**Conclusion.** LoRA recovery is essential.

### D.1.5 LayerNorm Recalibration

**Goal.** Add A3-style LN recalibration.

**Result.** <0.2% improvement, extra complexity.

**Conclusion.** Not worth it for ASVD.

## D.2 Abandoned Variants

### D.2.1 Tensor Train Decomposition

**Goal.** Use TT instead of SVD.

**Result.** 10× slower, no improvement.

### D.2.2 Mixed Ranks per Head

**Goal.** Different ranks per attention head.

**Result.** Complexity not worth it.

### D.2.3 Non-Linear SVD

**Goal.** Use kernel SVD.

**Result.** No improvement, harder to implement.

## D.3 Failed Hyperparameters

### Rank

| r | DistilBERT | Ratio |
|---|---|---|
| 4 | 72.0 | 24× |
| 8 | 80.5 | 12× |
| 16 | 86.0 | 8× |
| 32 | 87.5 | 5× |
| 64 | 88.5 | 3.5× |

### Energy Threshold

| $\varepsilon$ | DistilBERT |
|---|---|
| 0.85 | 82.0 |
| 0.90 | 87.5 |
| 0.95 | 88.5 |
| 0.99 | 89.5 |
| 0.999 | 89.9 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 88.5 |
| 123 | 88.4 |
| 456 | 88.6 |
| Mean | 88.5 |
| Std | 0.10 |

## D.5 Lessons Learned

1. **Diagonal weighting is sufficient.**
2. **r=16 is the practical minimum.**
3. **τ=0.20 is a good default.**
4. **LoRA recovery is essential.**
5. **LN recalibration not worth it.**

---