# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Very Small Base Rank (r_b = 4)

**Goal.** Maximize compression.

**Result.** Accuracy dropped by {{RB4_DROP}}%.

**Conclusion.** $r_b = 16$ is a practical minimum.

### D.1.2 Zero Residual Rank

**Goal.** Equivalent to pure SVD.

**Result.** Accuracy dropped by {{RR0_DROP}}%.

**Conclusion.** Residual rank > 0 is essential.

### D.1.3 Scaling α = 0.1

**Goal.** Small residual contribution.

**Result.** Accuracy dropped; residual was undertrained.

**Conclusion.** $\alpha = 1.0$ is a good default.

### D.1.4 Scaling α = 10

**Goal.** Large residual contribution.

**Result.** Unstable training.

**Conclusion.** $\alpha = 1.0$ works.

## D.2 Abandoned Variants

### D.2.1 Adaptively Grow Residual Rank

**Goal.** Learn rank during training.

**Result.** No convergence.

### D.2.2 Layer-Wise Base Rank

**Goal.** Different $r_b$ per layer.

**Result.** Marginal improvement, more complexity.

### D.2.3 Weighted Base SVD

**Goal.** Activation-weighted base.

**Result.** Same as A3; not the focus.

## D.3 Failed Hyperparameters

### Base Rank

| $r_b$ | DistilBERT |
|---|---|
| 4 | {{D_RB4}} |
| 8 | {{D_RB8}} |
| 16 | {{D_RB16}} |
| 32 | {{D_RB32}} |

### Residual Rank

| $r_r$ | DistilBERT |
|---|---|
| 0 | {{D_RR0}} |
| 2 | {{D_RR2}} |
| 4 | {{D_RR4}} |
| 8 | {{D_RR8}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| Mean | {{MEAN}} |
| Std | {{STD}} |

## D.5 Lessons Learned

1. **Residual rank > 0 is essential.**
2. **Base rank ≥ 16 for meaningful accuracy.**
3. **Scaling α = 1.0 is a good default.**
4. **Adaptive rank doesn't converge.**
5. **Small total rank = large accuracy drop.**

---