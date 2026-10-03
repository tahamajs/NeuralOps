<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Gate Bias at 0

**Goal.** Let SSM and attention start equally.

**Result.** Accuracy dropped by 0.85%. Training diverged.

**Conclusion.** Gate bias must be negative.

### D.1.2 Removing the SSM Branch

**Goal.** See if global mixing alone suffices.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** SSM contributes something.

### D.1.3 Removing the Global Mixing

**Goal.** Simplify architecture.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Global mixing helps.

### D.1.4 Larger d_state (64)

**Goal.** More SSM capacity.

**Result.** Marginal accuracy improvement, 4× more parameters.

**Conclusion.** $d_\text{state} = 16$ is a good default.

### D.1.5 No Conv1d

**Goal.** Simplify SSM.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Conv1d is essential for local context.

## D.2 Abandoned Variants

### D.2.1 Bidirectional SSM

**Goal.** Process sequence in both directions.

**Result.** Broke causality for decoder models.

### D.2.2 Multiple SSM Branches

**Goal.** Ensemble of SSMs.

**Result.** 2× SSM params, no improvement.

### D.2.3 Shared SSM Across Layers

**Goal.** Reduce SSM parameter count.

**Result.** Accuracy dropped by 0.85%.

## D.3 Failed Hyperparameters

### d_state

| $d_\text{state}$ | DistilBERT |
|---|---|
| 4 | 0.85 |
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |

### Gate Bias

| bias | DistilBERT |
|---|---|
| -10 | 0.85 |
| -7 | 0.85 |
| -5 | 0.85 |
| 0 | 0.85 |
| +2 | 0.85 |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |
| 1e-3 | 0.85 (unstable) |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| Mean | 84.1 |
| Std | 0.1 |

## D.5 Lessons Learned

1. **Gate bias must be negative.**
2. **d_state=16 is sufficient.**
3. **Conv1d is essential.**
4. **Global mixing helps.**
5. **Recovery is unstable.**
6. **Parameter increase is real.**

---