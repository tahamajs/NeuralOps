# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Decay (λ = 1)

**Goal.** Standard linear attention without decay.

**Result.** Accuracy dropped by 4.5%.

**Conclusion.** Decay is essential.

### D.1.2 Very Small λ ([-5, -3])

**Goal.** Very fast decay (only recent tokens matter).

**Result.** Loss convergence failed; accuracy collapsed.

**Conclusion.** λ must stay near 1 for stable training.

### D.1.3 No Chunking (single-pass cumsum)

**Goal.** Use cumsum over the full sequence.

**Result.** OOM at L > 512.

**Conclusion.** Chunking is essential for O(1) memory.

### D.1.4 Learned Feature Map

**Goal.** Use a small MLP as feature map.

**Result.** No improvement over ELU+1, slower.

**Conclusion.** ELU+1 is sufficient.

### D.1.5 Per-Token Chunk Size

**Goal.** Adaptive chunk size per token.

**Result.** Complex implementation, no benefit.

**Conclusion.** Fixed chunk size works.

## D.2 Abandoned Variants

### D.2.1 Non-Exponential Decay

Tested polynomial decay $1/t^k$. Result: worse accuracy.

### D.2.2 Learned λ Temperature

Tested learnable σ temperature for λ. Result: no improvement.

### D.2.3 Different Bidirectional Weights

Tested learnable weight for $O^\text{fwd}$ vs $O^\text{bwd}$. Result: 
converged to 0.5 anyway.

## D.3 Failed Hyperparameters

### Chunk Size

| $C$ | DistilBERT | State mem |
|---|---|---|
| 4 | 83.0 | 2 KB |
| 8 | 84.5 | 4 KB |
| 32 | 86.5 | 16 KB |
| 256 | 83.0 | 128 KB |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-4 | 83.5 |
| 2e-4 | 86.5 |
| 5e-4 | 82.0 |
| 1e-3 | NaN |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 86.5 |
| 123 | 86.3 |
| 456 | 86.7 |
| **Mean** | **86.5** |
| **Std** | **0.2** |

## D.5 Lessons Learned

1. **Decay is essential.**
2. **λ must be initialized near 1.**
3. **Chunking enables O(1) memory.**
4. **ELU+1 is sufficient.**
5. **Bidirectional average at 0.5 is fine.**

---