<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Very Small Sketch (r=4)

**Goal.** Maximize FLOPs reduction.

**Result.** Score correlation dropped to {{CORR4}}. Accuracy collapsed.

**Conclusion.** r=8 is the practical minimum.

### D.1.2 Shared Projection Across Heads

**Goal.** Use the same $\Omega$ for all heads.

**Result.** 0.5% accuracy drop, no speed benefit.

**Conclusion.** Head-wise is better.

### D.1.3 Learned Projection

**Goal.** Train $\Omega$ jointly with LoRA.

**Result.** Trained $\Omega$ lost the JL property, accuracy dropped.

**Conclusion.** $\Omega$ must be random and fixed.

### D.1.4 Sparse Random Projection (Achlioptas)

**Goal.** Use Achlioptas-style sparse projections.

**Result.** Similar accuracy, 10% faster sketch.

**Conclusion.** Marginal gain, complexity not worth it.

### D.1.5 Larger Scale Factor

**Goal.** Use $1/\sqrt{d_h}$ instead of $1/\sqrt{r}$.

**Result.** Softmax saturated, accuracy collapsed.

**Conclusion.** $1/\sqrt{r}$ is correct.

## D.2 Abandoned Variants

### D.2.1 Sketching V

**Goal.** Also sketch $V$.

**Result.** Output shape mismatch; problematic for downstream.

### D.2.2 Per-Token Projection

**Goal.** Different $\Omega$ per token.

**Result.** Not possible without breaking JL.

### D.2.3 Hierarchical Sketch

**Goal.** Multi-level sketches.

**Result.** Complexity not worth it.

## D.3 Failed Hyperparameters

### Sketch Rank

| $r$ | DistilBERT | Correlation |
|---|---|---|
| 4 | {{D_R4}} | {{CORR4}} |
| 8 | {{D_R8}} | {{CORR8}} |
| 16 | {{D_R16}} | {{CORR16}} |
| 32 | {{D_R32}} | {{CORR32}} |
| 64 | {{D_R64}} | 1.00 |

### LoRA Rank

| $r_L$ | DistilBERT |
|---|---|
| 0 | {{D_LORA0}} |
| 8 | {{D_LORA8}} |
| 16 | {{D_LORA16}} |
| 32 | {{D_LORA32}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| Mean | {{MEAN}} |
| Std | {{STD}} |

**Observation.** Seed variance is small (0.1%) because JL is 
concentration-of-measure.

## D.5 Lessons Learned

1. **r=8 is the practical minimum.**
2. **Head-wise projections are better.**
3. **Ω must be fixed (not learned).**
4. **Scale factor must be $1/\sqrt{r}$.**
5. **RPC does not reduce memory.**

## D.6 Reproducibility Note

The random seed for `proj` initialization affects results. We use seed 
{{SEED}} for all experiments.

---
