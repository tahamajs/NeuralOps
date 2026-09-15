<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 FP16 SVD

**Goal.** Compute SVD in FP16 to save memory.

**Result.** Occasional NaN for ill-conditioned matrices.

**Conclusion.** Use FP32 SVD.

### D.1.2 Random Calibration

**Goal.** Use random inputs instead of real text.

**Result.** Accuracy degraded to {{RANDOM_ACC}}%.

**Conclusion.** Real calibration required. Bundled corpus works.

### D.1.3 No Padding Filter

**Goal.** Skip padding token mask.

**Result.** Covariance biased; accuracy dropped by {{NO_FILTER_DROP}}%.

**Conclusion.** Always filter padding.

### D.1.4 Full-Matrix EMA on GPU

**Goal.** Keep covariance on GPU to avoid transfer.

**Result.** CUDA OOM on 12-layer models.

**Conclusion.** Must move to CPU.

### D.1.5 Diagonal-Only Weighting

**Goal.** Use only $\mathrm{diag}(C_X)$.

**Result.** Accuracy dropped by {{DIAG_DROP}}%.

**Conclusion.** Full covariance matters.

## D.2 Abandoned Variants

### D.2.1 Adaptive Rank

We tested adaptive rank escalation (like A3). Result: no improvement over 
fixed rank with the 0.75 cap.

### D.2.2 LayerNorm Recalibration

We tested adding A3-style LN recalibration. Result: <0.3% improvement, not 
worth the complexity.

### D.2.3 Nonlinear Covariance

We tested using $C_X^\alpha$ for $\alpha \neq 1/2$. Result: no improvement.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | {{D_LR1e5}} |
| 5e-5 | {{D_LR5e5}} |
| 2e-4 | {{D_LR2e4}} |
| 5e-4 | {{D_LR5e4}} (unstable) |

### Max Error

| τ | DistilBERT | Fallback % |
|---|---|---|
| 0.10 | {{D_T10}} | {{D_FB10}} |
| 0.20 | {{D_T20}} | {{D_FB20}} |
| 0.30 | {{D_T30}} | {{D_FB30}} |
| 0.50 | {{D_T50}} | {{D_FB50}} |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Different β init | Convergence | No change |
| Per-layer β | Non-stationarity | No change |
| SVD on CPU always | Stability | Slower, no gain |
| Separate calibration per layer | Precision | No change |

## D.5 Multi-Seed (Partial)

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Lessons Learned

1. **FP32 SVD is essential.**
2. **CPU transfer is required** for scaling.
3. **Full covariance > diagonal.**
4. **Bundled corpus is sufficient.**
5. **Frobenius fallback is critical.**
