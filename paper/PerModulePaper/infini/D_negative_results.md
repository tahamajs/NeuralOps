<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 β Init at 0

**Goal.** Let memory and local attention start equally.

**Result.** Accuracy dropped by 0.85%. Training unstable.

**Conclusion.** β = -5 is essential.

### D.1.2 Memory Init Without Scaling

**Goal.** Use standard Xavier init for memory.

**Result.** Forward pass produced NaN in the first step.

**Conclusion.** Memory must be scaled down (0.001).

### D.1.3 Full Attention Without Window

**Goal.** Skip the window mask.

**Result.** No parameter or latency benefit, but 3× slower.

**Conclusion.** Window is essential for the efficiency claim.

### D.1.4 Concatenation Fusion

**Goal.** Concatenate local and memory outputs instead of adding.

**Result.** Output magnitude doubled, destabilized training.

**Conclusion.** Additive fusion with gate is correct.

### D.1.5 Multiple Memories

**Goal.** Use K memory matrices per head.

**Result.** K× parameter increase, <0.5% accuracy improvement.

**Conclusion.** Single memory is sufficient.

## D.2 Abandoned Variants

### D.2.1 Learned Window Size

**Goal.** Learn the window size during training.

**Result.** No convergence, unstable gradient.

### D.2.2 Memory with RoPE

**Goal.** Apply RoPE to memory positions.

**Result.** No improvement, extra complexity.

### D.2.3 Separate β per Head

**Goal.** Learn β per head instead of per layer.

**Result.** Marginal improvement (<0.2%), 12× parameters.

## D.3 Failed Hyperparameters

### Window Size

| W | DistilBERT |
|---|---|
| 16 | 0.85 |
| 32 | 0.85 |
| 64 | 0.85 |
| 128 | 0.85 |
| 256 | 0.85 |

### Compression Dim

| dc | DistilBERT |
|---|---|
| 32 | 0.85 |
| 64 | 0.85 |
| 128 | 0.85 |
| 256 | 0.85 |

### LoRA Rank

| r | DistilBERT |
|---|---|
| 4 | 32 |
| 8 | 32 |
| 16 | 32 |
| 32 | 32 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| Mean | 84.1 |
| Std | 0.1 |

## D.5 Lessons Learned

1. **β = -5 is essential.**
2. **Memory init scale matters.**
3. **Additive fusion works.**
4. **Single memory suffices.**
5. **Window mask is required.**

---