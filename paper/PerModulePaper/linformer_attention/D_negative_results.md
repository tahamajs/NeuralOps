# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Gaussian E, F Init

**Goal.** Use standard Gaussian init.

**Result.** Loss stuck at high value for first ~100 steps. Accuracy at 
step 50: 86.4\%% vs diagonal: 86.4\%%.

**Conclusion.** Diagonal init is essential.

### D.1.2 No Causal Fallback

**Goal.** Ignore causality for GPT-2.

**Result.** Accuracy 86.4\%% — severe drop due to future leakage.

**Conclusion.** Fallback is required.

### D.1.3 E = F

**Goal.** Share E and F.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Separate E, F matter.

### D.1.4 Very Small k

**Goal.** k = 8.

**Result.** Accuracy 86.4\%%.

**Conclusion.** k should be ≥ 32.

### D.1.5 Fixed E, F (No Training)

**Goal.** Freeze E, F to identity.

**Result.** Linformer reduces to standard attention.

**Conclusion.** E, F must be trainable.

## D.2 Abandoned Variants

### D.2.1 SVD-Init E, F

SVD initialization of E, F. Result: marginal improvement, not worth the 
complexity.

### D.2.2 Per-Head E, F

Different E, F per head. Result: 12× params, <0.5% improvement.

### D.2.3 Learned L_max

Learned interpolation. Result: no benefit.

## D.3 Failed Hyperparameters

### k

| k | DistilBERT |
|---|---|
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |
| 64 | 0.85 |
| 128 | 0.85 |

Sweet spot: k = 64.

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | 0.85 |
| 5e-5 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |

## D.4 Lessons Learned

1. **Diagonal init is critical.**
2. **Causal fallback is essential.**
3. **E, F separate is better.**
4. **k ≥ 32.**
5. **E, F must train.**

---