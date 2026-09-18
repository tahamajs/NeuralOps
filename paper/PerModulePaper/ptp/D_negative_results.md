<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Without [CLS] Preservation

**Goal.** Prune tokens including [CLS] if it scores low.

**Result.** Accuracy collapsed to majority baseline.

**Conclusion.** [CLS] preservation is essential.

### D.1.2 Per-Query Top-K

**Goal.** Different T for each query.

**Result.** No improvement, 3× slower (loop).

**Conclusion.** Shared T suffices.

### D.1.3 Attention-Score Selection

**Goal.** Use attention magnitudes (from last layer) instead of MLP.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Learned MLP is better.

### D.1.4 Random Selection

**Goal.** Randomly keep $k$ tokens.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Selection matters.

### D.1.5 Merge Instead of Prune

**Goal.** Merge similar tokens.

**Result.** Slightly better accuracy but no memory savings.

**Conclusion.** Hard pruning for memory.

## D.2 Abandoned Variants

### D.2.1 Per-Head Scoring

Separate score network per head. No improvement.

### D.2.2 Learned Keep Ratio

Learn $\rho$ via a global logit. Unstable.

### D.2.3 Recursive Pruning

Prune multiple times per layer. No benefit.

## D.3 Failed Hyperparameters

### Keep Ratio Too Low
| $\rho$ | DistilBERT |
|---|---|
| 0.3 | 32 |
| 0.4 | 32 |
| 0.5 | 32 |

### Score Network Too Small
| Hidden | DistilBERT |
|---|---|
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Different activation | Quality | No improvement |
| Deeper MLP | Quality | No improvement |
| Positional embedding | Quality | No improvement |
| Learned mask | Quality | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Lessons Learned

1. **[CLS] preservation is essential.**
2. **Shared selection suffices.**
3. **Learned MLP beats heuristics.**
4. **Keep ratio 0.8 is a sweet spot.**
5. **MLP-64 suffices.**

---