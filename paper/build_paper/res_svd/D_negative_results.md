# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Very Small Base Rank (r_b = 4)

**Goal.** Maximize compression.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** $r_b = 16$ is a practical minimum.

### D.1.2 Zero Residual Rank

**Goal.** Equivalent to pure SVD.

**Result.** Accuracy dropped by 0.85%.

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
| 4 | 32 |
| 8 | 32 |
| 16 | 32 |
| 32 | 32 |

### Residual Rank

| $r_r$ | DistilBERT |
|---|---|
| 0 | 32 |
| 2 | 32 |
| 4 | 32 |
| 8 | 32 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| Mean | 84.1 |
| Std | 0.1 |

## D.5 Lessons Learned

1. **Residual rank > 0 is essential.**
2. **Base rank ≥ 16 for meaningful accuracy.**
3. **Scaling α = 1.0 is a good default.**
4. **Adaptive rank doesn't converge.**
5. **Small total rank = large accuracy drop.**

---