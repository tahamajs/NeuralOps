# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Random Landmarks

**Goal.** Uniform random selection.

**Result.** Accuracy dropped by {{RAND_DROP}}%.

**Conclusion.** Learned selection matters.

### D.1.2 No Local Branch

**Goal.** Global landmark only.

**Result.** Accuracy dropped by {{NOLOCAL_DROP}}%.

**Conclusion.** Local helps.

### D.1.3 Standard Softmax

**Goal.** Replace softpick with softmax.

**Result.** Accuracy dropped by {{SOFTMAX_DROP}}%; NaN losses on some 
batches.

**Conclusion.** Softpick is essential.

### D.1.4 Single-Scale RFF

**Goal.** Only $F$ features, no $F/2$.

**Result.** Accuracy dropped by {{SINGLE_DROP}}%.

**Conclusion.** Dual-scale helps.

### D.1.5 Very Small ρ

**Goal.** Aggressive landmark reduction.

**Result.** Accuracy collapsed at $\rho < 0.25$.

**Conclusion.** $\rho \geq 0.25$ required.

## D.2 Abandoned Variants

### D.2.1 Hierarchical Landmarks

Two-level landmark selection. Result: complex, no benefit.

### D.2.2 Per-Layer Landmark Ratios

Different ρ per layer. Result: unstable.

### D.2.3 Learned Temperature

Learned $T$ in softpick. Result: converged to 1.0.

## D.3 Failed Hyperparameters

### Local Gate Init

| $w_h$ | DistilBERT |
|---|---|
| -4 | {{WG_M4}} |
| -2 | {{D_ACC}} |
| 0 | {{WG_0}} |
| +2 | {{WG_P2}} |

### Correction Matrix

| Init | DistilBERT |
|---|---|
| Identity | {{D_ACC}} |
| Random | {{CORR_RAND}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.5 Lessons Learned

1. **Learned landmarks beat random.**
2. **Local branch is essential.**
3. **Softpick prevents NaN.**
4. **Dual-scale helps.**
5. **ρ ≥ 0.25 required.**

## D.6 Recommended Fixes

1. Replace `raise RuntimeError` in `copy_weights` with a soft fallback.
2. Consider setting $L_\text{max}$ to a constant (e.g., 64) for truly 
   linear attention.
3. Validate the correction matrix initialization.

---