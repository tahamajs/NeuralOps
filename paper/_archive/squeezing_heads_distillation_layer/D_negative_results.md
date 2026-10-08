<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Head Pruning Instead of Dim Reduction

**Goal.** Prune entire heads instead of reducing head dimension.

**Result.** Accuracy dropped by 5% — more than dim reduction.

**Conclusion.** Head-dim reduction preserves more information.

### D.1.2 Random Head-Dim Slice

**Goal.** Slice random head dimensions instead of leading.

**Result.** Accuracy dropped by 2.5%.

**Conclusion.** Leading dimensions are more informative.

### D.1.3 No LoRA Recovery

**Goal.** Apply head-dim reduction without LoRA.

**Result.** Accuracy dropped by 6.5%.

**Conclusion.** LoRA is essential for recovery.

### D.1.4 Per-Head LoRA

**Goal.** Separate LoRA adapter per head.

**Result.** No improvement over shared LoRA; more parameters.

**Conclusion.** Shared LoRA suffices.

### D.1.5 Full Fine-Tuning

**Goal.** Train full model after head-dim reduction.

**Result.** Similar to LoRA but 100× more compute.

**Conclusion.** LoRA is more efficient.

## D.2 Abandoned Variants

### D.2.1 Learned Head-Dim

Instead of fixed reduction, learn per-head dimension. Complexity not worth it.

### D.2.2 Adaptive LoRA Rank

Per-layer rank allocation. Marginal improvement.

### D.2.3 Teacher Distillation

Distill from larger model. Orthogonal to our method.

## D.3 Failed Hyperparameters

### LoRA Rank Too Small
| $r_L$ | DistilBERT |
|---|---|
| 1 | 82.0 |
| 2 | 85.0 |
| 4 | 88.5 |

### Head-Dim Reduction Too Large
| $s$ | DistilBERT |
|---|---|
| 3.0 | 78.0 |
| 4.0 | 65.0 |

### Recovery Too Short
| Steps | DistilBERT |
|---|---|
| 5 | 78.0 |
| 10 | 85.0 |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Per-head LoRA | Quality | No improvement |
| Adaptive rank | Quality | No improvement |
| Random slice | Quality | Worse |
| Weight decay tuning | Quality | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 90.25 |
| 123 | 90.10 |
| 456 | 90.35 |
| **Mean** | **90.23** |
| **Std** | **0.13** |

## D.6 Lessons Learned

1. **Head-dim reduction > head pruning.**
2. **Leading dimensions are best.**
3. **LoRA is essential.**
4. **Rank 16 suffices.**
5. **Shared LoRA is sufficient.**

---
