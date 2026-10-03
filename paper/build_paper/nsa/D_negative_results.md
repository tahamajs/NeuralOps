<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Top-K Before Causal

**Goal.** Compute top-$k$ without causal mask.

**Result.** Decoder outputs leaked future information.

**Conclusion.** Always mask before top-$k$.

### D.1.2 Top-K Before Local Mask

**Goal.** Compute top-$k$ on all scores, then add local window.

**Result.** Top-$k$ selections duplicated window entries; wasted budget.

**Conclusion.** Mask local window before top-$k$.

### D.1.3 Learned Top-K

**Goal.** Learn $k$ per head via gradient.

**Result.** Non-differentiable; unstable.

**Conclusion.** Keep $k$ fixed.

### D.1.4 Adaptive Window

**Goal.** Learn window size per head.

**Result.** No improvement; complexity.

**Conclusion.** Fixed window.

### D.1.5 Clustering-Based Selection

**Goal.** Replace top-$k$ with K-means selection.

**Result.** Slower, no improvement.

**Conclusion.** Top-$k$ is sufficient.

## D.2 Abandoned Variants

### D.2.1 Two-Sided Top-K

Select top-$k$ on both past and future. Not applicable to causal.

### D.2.2 Sink-Preserved Top-K

Force first token always selected. Marginal improvement.

### D.2.3 Per-Layer Window

Different $w$ per layer. Marginal improvement.

## D.3 Failed Hyperparameters

### Window Too Small

| $w$ | DistilBERT |
|---|---|
| 2 | 0.85 |
| 4 | 0.85 |
| 8 | 0.85 |
| 16 | 0.85 |

### Top-K Too Small

| $k$ | DistilBERT |
|---|---|
| 1 | 0.85 |
| 2 | 0.85 |
| 4 | 0.85 |

### Density Too High

Even at density 100% (large $w + k$), accuracy doesn't exceed baseline 
(because the mask is trivial).

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Learned selection threshold | Quality | No improvement |
| Two-pass selection | Quality | Slower |
| Sparse tensor mask | Memory | Slower in Python |
| Cached top-k indices | Speed | Not applicable at inference |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Lessons Learned

1. **Mask order matters:** local → causal → top-$k$.
2. **Short sequences skip top-$k$.**
3. **Fixed $w, k$ suffices.**
4. **No time speedup without sparse kernels.**
5. **Density decreases with $L$.**

---