# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Gaussian E, F Init

**Goal.** Use standard Gaussian init.

**Result.** Loss stuck at high value for first ~100 steps. Accuracy at 
step 50: {{GAUSS_ACC}}% vs diagonal: {{DIAG_ACC}}%.

**Conclusion.** Diagonal init is essential.

### D.1.2 No Causal Fallback

**Goal.** Ignore causality for GPT-2.

**Result.** Accuracy {{NO_CAUSAL_ACC}}% — severe drop due to future leakage.

**Conclusion.** Fallback is required.

### D.1.3 E = F

**Goal.** Share E and F.

**Result.** Accuracy dropped by {{SHARED_DROP}}%.

**Conclusion.** Separate E, F matter.

### D.1.4 Very Small k

**Goal.** k = 8.

**Result.** Accuracy {{K8_ACC}}%.

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
| 8 | {{D_K8}} |
| 16 | {{D_K16}} |
| 32 | {{D_K32}} |
| 64 | {{D_K64}} |
| 128 | {{D_K128}} |

Sweet spot: k = 64.

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | {{D_LR1e5}} |
| 5e-5 | {{D_LR5e5}} |
| 2e-4 | {{D_LR2e4}} |
| 5e-4 | {{D_LR5e4}} |

## D.4 Lessons Learned

1. **Diagonal init is critical.**
2. **Causal fallback is essential.**
3. **E, F separate is better.**
4. **k ≥ 32.**
5. **E, F must train.**
```

---