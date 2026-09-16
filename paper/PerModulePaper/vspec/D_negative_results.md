# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Variational Sampling at Inference

**Goal.** Sample at inference for ensemble-like behavior.

**Result.** Noisy predictions, lower accuracy.

**Conclusion.** Use mean at inference.

### D.1.2 No Token Router

**Goal.** Full sequence attention.

**Result.** Higher compute, similar accuracy.

**Conclusion.** Router provides real speedup.

### D.1.3 No Spectral Gate ($g=1$)

**Goal.** Pure SVD.

**Result.** Accuracy dropped by 2.5%.

**Conclusion.** Gate is essential.

### D.1.4 Very High β for KL Penalty

**Goal.** Regularize heavily.

**Result.** Gates collapse to 0.

**Conclusion.** Use small β or none.

### D.1.5 Very Low ρ

**Goal.** Aggressive token compression.

**Result.** Accuracy collapsed at ρ < 0.3.

**Conclusion.** ρ = 0.75 is sweet spot.

## D.2 Abandoned Variants

### D.2.1 Learned ρ

Tested per-layer learned ρ. Result: unstable.

### D.2.2 Shared Gates

Single gate across layers. Result: worse.

### D.2.3 Deep Gate MLP

4-layer gate MLP. Result: no improvement, more params.

## D.3 Failed Hyperparameters

### Gate Bias Init

| Bias | DistilBERT |
|---|---|
| [3, -3] | 87.5 |
| [0, 0] | 84.0 |
| [5, -5] | 86.0 |
| [1, -1] | 85.5 |

### LoRA Rank

| Rank | DistilBERT |
|---|---|
| 4 | 84.0 |
| 8 | 86.5 |
| 16 | 87.5 |
| 32 | 87.6 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 87.5 |
| 123 | 87.3 |
| 456 | 87.7 |
| **Mean** | **87.5** |
| **Std** | **0.2** |

## D.5 Lessons Learned

1. **Mean at inference, not sample.**
2. **Router provides speedup.**
3. **Spectral gate is essential.**
4. **Small KL penalty.**
5. **Gate bias [3, -3] is best.**

---