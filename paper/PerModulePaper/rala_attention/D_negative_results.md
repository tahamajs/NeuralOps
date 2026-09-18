<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 What Didn't Work

### D.1.1 Frozen $U, V$ is a Limitation

**Hypothesis.** Freezing $U, V$ and training only $\alpha$ would be enough.

**Result.** The augmentation is a random projection. Accuracy improvement 
over pure linear attention is marginal.

**Conclusion.** Future work should unfreeze $U, V$ at least at the 
last few layers.

### D.1.2 Scalar Gate Too Simple

**Hypothesis.** A single scalar $\alpha$ per layer is sufficient.

**Result.** Different heads may need different augmentation strengths.

**Conclusion.** Per-head gating would be better.

### D.1.3 Softplus vs elu+1

**Hypothesis.** Softplus would be measurably better than elu+1.

**Result.** Difference is within noise: 0.85%.

**Conclusion.** Softplus is not clearly better in practice.

### D.1.4 Sequential Scan

**Hypothesis.** The Python loop over sequence length would not be a 
bottleneck.

**Result.** It dominates runtime — up to 48.5\%% of forward pass.

**Conclusion.** Fused parallel scan is essential for production.

## D.2 Failed Hyperparameters

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |

## D.3 Abandoned Variants

### D.3.1 Per-Head Alpha

We tried $H$ scalars per layer. Result: modest accuracy improvement (~0.5%), 
larger parameter count. Not adopted.

### D.3.2 Rank-1 Augmentation

We tried $r = 1$ (single rank). Result: minimal effect. Not adopted.

### D.3.3 Unfrozen $U, V$

We tried unfreezing $U, V$. Result: loss oscillated, did not converge. 
Requires more tuning.

## D.4 Lessons Learned

1. **Softplus is not a magic kernel.**
2. **The sequential scan is a fatal flaw for production.**
3. **Frozen $U, V$ limits expressiveness.**
4. **A scalar gate is minimal but usable.**

## D.5 Recommendations

1. **Fuse the prefix scan** before any further experiments.
2. **Unfreeze $U, V$** for at least some layers.
3. **Try per-head gates** if tuning compute allows.
4. **Benchmark against Delta Net, GLA, Mamba** for context.

---