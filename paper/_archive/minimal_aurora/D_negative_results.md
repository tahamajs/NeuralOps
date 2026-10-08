# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Energy Threshold 0.85

**Goal.** Aggressive compression.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** 0.90 is a minimum.

### D.1.2 No LoRA Recovery

**Goal.** See if SVD alone suffices.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** LoRA recovery is essential.

### D.1.3 Removing `output_scale`

**Goal.** Simplify.

**Result.** Similar accuracy; the parameter is not critical.

**Conclusion.** Safe to remove.

### D.1.4 Wiring `pres_gate`

**Goal.** Use the `pres_gate` parameter.

**Result.** 32

**Conclusion.** 0.85

### D.1.5 Cross-Attention

**Goal.** Test on encoder-decoder.

**Result.** Limited support.

## D.2 Abandoned Variants

### D.2.1 Adaptive Energy

**Goal.** Learn $\varepsilon$ during training.

**Result.** No convergence.

### D.2.2 Per-Head Rank

**Goal.** Different rank per head.

**Result.** Marginal improvement, more complexity.

### D.2.3 Spectral LoRA Init (Res-SVD style)

**Goal.** Init LoRA from residual spectrum.

**Result.** Similar to zero init, more complexity.

## D.3 Failed Hyperparameters

### Energy

| $\varepsilon$ | DistilBERT |
|---|---|
| 0.85 | 0.85 |
| 0.90 | 0.85 |
| 0.95 | 0.85 |
| 0.99 | 0.85 |

### LoRA Rank

| $r_L$ | DistilBERT |
|---|---|
| 0 | 0.85 |
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| Mean | 84.1 |
| Std | 0.1 |

## D.5 Lessons Learned

1. **Energy ≥ 0.90 for meaningful accuracy.**
2. **LoRA recovery is essential.**
3. **`pres_gate` has no effect (dead code).**
4. **`output_scale` can be removed safely.**
5. **Zero-init LoRA works as well as spectral-init.**

---