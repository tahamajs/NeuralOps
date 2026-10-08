<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 What Didn't Work

### D.1.1 No Inference Speedup

**Hypothesis.** With SALT, inference would be 2-4× faster due to low-rank.

**Result.** Naive forward is **slower** (2.3 s×). Reconstruction 
overhead dominates.

**Conclusion.** Need fused forward.

### D.1.2 γ Fixed at 1

**Hypothesis.** SVD truncation alone is sufficient; γ provides no benefit.

**Result.** Accuracy drops by 0.85% without trainable γ.

**Conclusion.** γ helps.

### D.1.3 AB = 0 (Frozen)

**Hypothesis.** SVD only is enough.

**Result.** Accuracy drops by 0.85%.

**Conclusion.** Residual AB is essential.

### D.1.4 Rank 4

**Hypothesis.** r=4 is sufficient.

**Result.** Accuracy drops significantly (32%).

**Conclusion.** r=16 is a good minimum.

## D.2 Failed Hyperparameters

| Rank | DistilBERT | Comment |
|---|---|---|
| 2 | 32 | Too aggressive |
| 4 | 32 | Marginal |
| 8 | 32 | OK |
| 16 | 32 | Default |
| 32 | 32 | Overkill |
| 64 | 32 | No benefit |

## D.3 Bugs Found During Development

### D.3.1 Unused `output_scale`

Defined but never used. Should be removed.

### D.3.2 `raise RuntimeError` in `copy_weights`

Aborts injection on any extraction error. Should be a warning.

### D.3.3 `raise RuntimeError` on Zero Norm

Same issue.

### D.3.4 GQA Pooling Loses Information

When source has more KV heads, we average. This is lossy.

## D.4 Abandoned Variants

### D.4.1 Trainable U, V

We tried unfreezing U and V. Result: loss oscillated, base weights drifted 
arbitrarily.

**Conclusion.** Freeze the SVD skeleton.

### D.4.2 Per-Channel Gamma

We tried per-output-channel γ instead of per-singular-value. Result: more 
parameters, no clear benefit.

### D.4.3 Different Ranks for SVD and Residual

We tried $r_\text{SVD} = 8, r_\text{residual} = 16$. Result: no clear 
benefit.

## D.5 Lessons Learned

1. **Fused forward is essential for inference speedup.**
2. **Both γ and AB contribute.**
3. **Rank 16 is a good balance.**
4. **Freezing SVD buffers is important.**
5. **Raising exceptions is bad practice.**

## D.6 Recommendations

1. **Implement fused forward.**
2. **Replace `raise` with warnings.**
3. **Remove dead `output_scale`.**
4. **Consider SALT + INT8 combination.**
5. **Add unit tests.**

---