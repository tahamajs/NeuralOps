<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Fixed Gate at 0.5

**Goal.** Equal mix of dense and linear.

**Result.** Accuracy dropped by 3.5%.

**Conclusion.** Learned gates are essential.

### D.1.2 Gate Bias = 0

**Goal.** Start balanced.

**Result.** Model drifted toward linear; accuracy dropped.

**Conclusion.** Start dense-biased.

### D.1.3 ReLU Kernel

**Goal.** Simpler kernel.

**Result.** Not smooth; unstable gradients.

**Conclusion.** ELU+1 better.

### D.1.4 Identity Kernel

**Goal.** Pure linear (no nonlinearity).

**Result.** Accuracy collapsed (non-positive features).

**Conclusion.** Nonlinear kernel essential.

### D.1.5 Chunked Cumsum

**Goal.** Reduce cumsum memory.

**Result.** Slower in Python implementation.

**Conclusion.** Needs Triton kernel.

## D.2 Abandoned Variants

### D.2.1 Per-Layer Gate

Instead of per-head, use per-layer gate. Less expressive.

### D.2.2 Softmax Gate

Instead of sigmoid, use softmax over both streams. No improvement.

### D.2.3 Multi-Stream (3+ branches)

Add local window as third branch. Marginal improvement.

## D.3 Failed Hyperparameters

### Gate Bias Too Low
| $b_h$ | DistilBERT |
|---|---|
| 0 | 85.0 |
| 0.5 | 87.0 |
| 1.0 | 88.5 |
| 1.8 | 89.5 |

### Bias Too High
| $b_h$ | DistilBERT |
|---|---|
| 3.0 | 89.0 |
| 5.0 | 88.0 |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Chunked cumsum (Python) | Speed | Slower |
| Learned kernel | Quality | No improvement |
| Per-token gate | Quality | Unstable |
| Gate gradient clipping | Stability | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 89.5 |
| 123 | 89.3 |
| 456 | 89.7 |
| **Mean** | **89.5** |
| **Std** | **0.2** |

## D.6 Lessons Learned

1. **Learned gates beat fixed.**
2. **Dense bias at init is essential.**
3. **ELU+1 is the best kernel.**
4. **Cumsum needs FP32.**
5. **No speedup without hard routing.**

---
