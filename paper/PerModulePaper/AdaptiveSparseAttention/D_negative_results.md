<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Hard Top-$k$

**Goal.** Replace soft mask with hard top-$k$.

**Result.** Zero gradient; training diverged.

**Conclusion.** Soft mask is necessary for end-to-end training.

### D.1.2 Gumbel Softmax

**Goal.** Use Gumbel-softmax for stochastic sampling.

**Result.** High variance; unstable.

**Conclusion.** Deterministic sigmoid mask is better.

### D.1.3 Learned Chunk Size

**Goal.** Learn chunk size via gradient.

**Result.** No clear optimum; stuck at initial value.

**Conclusion.** Chunk size remains a hyperparameter.

### D.1.4 Per-Layer Sparsity Logits

**Goal.** Use different logits per (layer, head) pair.

**Result.** No improvement over per-head logits.

**Conclusion.** Per-head is sufficient.

### D.1.5 Per-Query Sparsity

**Goal.** Different $k_h$ per query position.

**Result.** Extra parameters; no improvement.

**Conclusion.** Per-head uniform budget is best.

## D.2 Abandoned Variants

### D.2.1 Learned Landmarks

Learned sampling of keys instead of top-$k$. Slower, no improvement.

### D.2.2 Content-Aware Threshold

Learned threshold instead of top-$k$ position. No improvement.

### D.2.3 Multi-Stage Sparsity

Sparsity at multiple attention layers. Marginal improvement.

## D.3 Failed Hyperparameters

### Temperature Too Small
| $T$ | DistilBERT | Note |
|---|---|---|
| 0.001 | {{D_T001}} | Near hard top-k |
| 0.01 | {{D_T01}} | Very aggressive |
| 0.1 | {{D_T1}} | Best |
| 1.0 | {{D_T10}} | Too soft |

### Chunk Size Too Small
| $C$ | DistilBERT | Time |
|---|---|---|
| 8 | {{D_C8}} | {{D_C8_T}} |
| 16 | {{D_C16}} | {{D_C16_T}} |
| 32 | {{D_C32}} | {{D_C32_T}} |
| 64 | {{D_C64}} | {{D_C64_T}} |

Very small chunks add overhead.

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Straight-through hard mask | Gradient | Unstable |
| Learned temperature | Quality | No improvement |
| Per-head K/V | Quality | No improvement |
| Multi-chunk gradients | Quality | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Lessons Learned

1. **Soft mask is essential** for differentiability.
2. **$T = 0.1$** is the optimal temperature.
3. **Per-head logits suffice;** per-layer not needed.
4. **Chunk size remains a hyperparameter.**
5. **Learned budgets are interpretable and bimodal.**
```

---