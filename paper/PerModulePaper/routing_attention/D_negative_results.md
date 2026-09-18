<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 What Didn't Work

### D.1.1 No Speedup with Naive Loop

**Hypothesis.** Clustering would reduce compute.

**Result.** The naive loop over clusters computes $K$ full attention 
matrices. Runtime **increased** by 2.3 s×.

**Conclusion.** Sparse matmul kernels are required.

### D.1.2 k-Means with Padding

**Hypothesis.** Padding tokens don't affect clustering.

**Result.** Padding distorts centroids, especially for short sequences.

**Conclusion.** Exclude padding before clustering.

### D.1.3 Q-Only Clustering

**Hypothesis.** Q alone suffices for clustering.

**Result.** Accuracy drops by 0.85%.

**Conclusion.** Q+K is better.

### D.1.4 No Token-0 Anchor

**Hypothesis.** Anchor is unnecessary.

**Result.** Accuracy drops by 0.85%.

**Conclusion.** Anchor is essential.

## D.2 Failed Hyperparameters

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |

## D.3 Abandoned Variants

### D.3.1 Random Init k-Means

We tried k-means++ style random init. Result: high variance, no clear 
benefit over deterministic.

### D.3.2 Per-Query Top-k

We tried per-query top-$k$ keys instead of clustering. Result: non-differentiable, 
slower.

### D.3.3 Soft Clustering

We tried Gumbel-Softmax for soft cluster assignment. Result: loss did not 
converge within 50 steps.

## D.4 RuntimeError on Zero Norm

Our `copy_weights` **raises** on zero norm:

```python
if w_norm < 1e-4:
    raise RuntimeError(...)
```

This aborts injection for the entire model. Should be a warning.

## D.5 Missing Double-Injection Guard

Running `inject_routing` twice corrupts the model.

## D.6 Lessons Learned

1. **Naive sparse attention is slower, not faster.**
2. **Padding must be excluded from clustering.**
3. **Q+K beats Q.**
4. **Token-0 anchor is essential.**
5. **k-means init should be deterministic.**

## D.7 Recommendations

1. **Implement sparse matmul before deploying.**
2. **Exclude padding.**
3. **Use Q+K.**
4. **Keep token-0 anchor.**
5. **Fix RuntimeError.**

---