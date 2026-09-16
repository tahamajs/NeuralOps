# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Broken Softpick (V4)

**Goal.** Use the original ReLU-based softpick.

**Result.** Attention collapsed to all-zero on masked rows.

**Conclusion.** Replaced with stable softmax in V5.

### D.1.2 copy_weights Raising

**Goal.** Abort on weight extraction failure.

**Result.** Many layers silently kept the original attention.

**Conclusion.** V5 uses warning + fallback.

### D.1.3 Full Rank SVD

**Goal.** Use `max_rank=384` (energy 0.95).

**Result.** No parameter reduction.

**Conclusion.** Use `max_rank < d/2`.

### D.1.4 No Decoder Fallback

**Goal.** Use split-head for GPT-2.

**Result.** Causality broken → accuracy collapse.

**Conclusion.** Fallback essential.

### D.1.5 All Global Heads

**Goal.** Use only kernel linear attention.

**Result.** Accuracy dropped significantly.

**Conclusion.** Local heads are essential.

## D.2 Abandoned Variants

### D.2.1 Sinkhorn Landmark Attention

Tested Sinkhorn normalization. No improvement.

### D.2.2 Per-Layer SVD Ranks

Learned per-layer ranks. Marginal improvement, complexity not worth it.

### D.2.3 Adaptive Window Size

Learned window size per layer. No benefit.

## D.3 Failed Hyperparameters

### Energy

| $\eta$ | DistilBERT |
|---|---|
| 0.80 | {{D_E80}} |
| 0.90 | {{D_E90}} |
| 0.95 | {{D_E95}} |
| 0.99 | {{D_E99}} |

Sweet spot: 0.95.

### Gate Bias

| Bias | DistilBERT |
|---|---|
| 0.0 | {{D_GB0}} |
| 1.0 | {{D_GB1}} |
| 2.0 | {{D_GB2}} |
| 4.0 | {{D_GB4}} |

Sweet spot: 2.0.

## D.4 Lessons Learned

1. **Stable softmax is essential.**
2. **Warning + fallback > abort.**
3. **`max_rank < d/2` for real compression.**
4. **Decoder fallback matters.**
5. **Split-head helps.**
```

---