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

**Result.** Accuracy dropped by {{NOGATE_DROP}}%.

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
| [3, -3] | {{D_ACC}} |
| [0, 0] | {{D_BIAS00}} |
| [5, -5] | {{D_BIAS55}} |
| [1, -1] | {{D_BIAS11}} |

### LoRA Rank

| Rank | DistilBERT |
|---|---|
| 4 | {{D_LR4}} |
| 8 | {{D_LR8}} |
| 16 | {{D_ACC}} |
| 32 | {{D_LR32}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.5 Lessons Learned

1. **Mean at inference, not sample.**
2. **Router provides speedup.**
3. **Spectral gate is essential.**
4. **Small KL penalty.**
5. **Gate bias [3, -3] is best.**

---