# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Causal Block-Sparse

**Goal.** Block-sparse for GPT-2.

**Result.** Causality broken, accuracy collapse.

**Conclusion.** Fallback to dense.

### D.1.2 Stripe at $N=128$

**Goal.** Force stripe for short sequences.

**Result.** Accuracy dropped significantly.

**Conclusion.** Gate stripe at $N > 256$.

### D.1.3 RFF Without Rectification Gate

**Goal.** Always apply RFF.

**Result.** Degraded encoder accuracy.

**Conclusion.** Periodic application is better.

### D.1.4 Softpick Without Clamp

**Goal.** Remove `clamp(max=85)`.

**Result.** NaN from `exp(large)`.

**Conclusion.** Clamp is essential.

### D.1.5 No CLS Boost

**Goal.** Don't force block 0.

**Result.** Classifier accuracy dropped by 0.85%.

**Conclusion.** Block-0 preservation is essential.

## D.2 Abandoned Variants

### D.2.1 Adaptive Block Size

Different block sizes per layer. No improvement.

### D.2.2 Multiple Seer Gates

Stacked Seer networks. No improvement.

### D.2.3 Learnable Softpick Temperature

Learned T. Marginal improvement.

## D.3 Failed Hyperparameters

### Block Size

| $B_s$ | DistilBERT |
|---|---|
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |
| 64 | 0.85 |

Sweet spot: 16.

### Sparse Min Seq Len

| min | DistilBERT |
|---|---|
| 4 | 84.5 |
| 8 | 84.5 |
| 16 | 84.5 |
| 32 | 84.5 |

Sweet spot: 8.

## D.4 Lessons Learned

1. **Causal block-sparse is hard.**
2. **Stripe needs long sequences.**
3. **RFF is inactive by default.**
4. **Clamping prevents NaN.**
5. **Block-0 preservation matters.**

---