# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Aux Loss

**Goal.** Remove aux loss to see if routers self-balance.

**Result.** Router collapsed to 1-2 experts; accuracy dropped to 86.4\%%.

**Conclusion.** Aux loss is essential.

### D.1.2 No Noise

**Goal.** Deterministic routing.

**Result.** Training unstable for first 20 steps.

**Conclusion.** Noise helps early training.

### D.1.3 Dense Experts (No SVD)

**Goal.** Compare with dense experts.

**Result.** Params 8× original; OOM on 12-layer models.

**Conclusion.** SVD is necessary for parameter control.

### D.1.4 E = 16

**Goal.** More experts.

**Result.** Params 3× original; OOM.

**Conclusion.** E = 8 is a practical upper limit on 8 GB GPU.

### D.1.5 Top-k = 4

**Goal.** More experts per token.

**Result.** 4× FLOPs; no accuracy improvement.

**Conclusion.** k = 2 is optimal.

## D.2 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 (unstable) |

### SVD Rank

| Rank | DistilBERT |
|---|---|
| d/8 | 86.4\% |
| d/4 | 86.4\% |
| d/2 | 86.4\% |
| d | 86.4\% |

## D.3 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Larger noise | Exploration | Unstable |
| Different init | Stability | No change |
| Shared experts | Parameter reduction | Accuracy drop |
| Hierarchical MoE | Specialization | No improvement |

## D.4 Lessons Learned

1. **Zero-init is critical** for stability.
2. **Aux loss is essential** for balance.
3. **E = 8 is the practical maximum** on 8 GB.
4. **SVD reduces bloat** but does not eliminate it.
5. **k = 2 is optimal** for SST-2.

## D.5 Recommendations

1. **Use E = 8, k = 2.**
2. **Set λ_aux = 0.01.**
3. **Enable noise with σ = 0.05.**
4. **Zero-init final expert layer.**
5. **Enable SVD experts** for parameter control.

---