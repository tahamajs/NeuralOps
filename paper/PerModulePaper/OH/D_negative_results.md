<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Gate Bias at 0

**Goal.** Let local and global branches start equally.

**Result.** Accuracy dropped by 4.2%. Training unstable.

**Conclusion.** Bias 2.0 is essential.

### D.1.2 No Compression Mask

**Goal.** Skip padding mask in compression softmax.

**Result.** Padding tokens polluted the compressed summary. Accuracy 
dropped by 3.5%.

**Conclusion.** Masking is essential.

### D.1.3 Concatenation Fusion

**Goal.** Concatenate local and global outputs.

**Result.** Output dimension doubled, destabilized.

**Conclusion.** Gated fusion is correct.

### D.1.4 Mean-Pool Compression

**Goal.** Use mean-pool instead of softmax-attention.

**Result.** Lost 1.5% accuracy.

**Conclusion.** Softmax-attention compression is better.

### D.1.5 Fixed Compression Rank

**Goal.** Use the same $d_c$ for all layers.

**Result.** Suboptimal. Some layers need more, some less.

**Conclusion.** Our uniform setting is a simplification.

## D.2 Abandoned Variants

### D.2.1 Per-Layer $d_c$

**Goal.** Different $d_c$ per layer.

**Result.** 20% overhead for <0.5% gain.

### D.2.2 Learned Window Size

**Goal.** Learn $W$ during training.

**Result.** No convergence.

### D.2.3 Memory + Compression

**Goal.** Combine Infini-Attention's memory with OHA's compression.

**Result.** 2× parameters, no improvement.

## D.3 Failed Hyperparameters

### Window Size

| W | DistilBERT |
|---|---|
| 16 | 87.0 |
| 32 | 88.5 |
| 64 | 89.2 |
| 128 | 89.4 |

### Compression Rank

| dc | DistilBERT |
|---|---|
| 8 | 87.5 |
| 16 | 88.5 |
| 32 | 89.2 |
| 64 | 89.3 |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 88.5 |
| 2e-4 | 89.2 |
| 5e-4 | 87.0 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 89.2 |
| 123 | 89.1 |
| 456 | 89.3 |
| Mean | 89.2 |
| Std | 0.08 |

## D.5 Lessons Learned

1. **Gate bias 2.0 is essential.**
2. **Compression masking is essential.**
3. **Gated fusion beats concatenation.**
4. **Softmax-attention beats mean-pool.**
5. **`d_c = 32` is a good default.**

---
