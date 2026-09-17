<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Hard Window Mask

**Goal.** Replace soft ramp with hard window.

**Result.** Zero gradient w.r.t. span; training diverged.

**Conclusion.** Soft ramp is necessary for differentiability.

### D.1.2 Initial Span Small

**Goal.** Initialize span to $S_\text{max}/4$ to save compute from start.

**Result.** Accuracy dropped from epoch 0.

**Conclusion.** Start fully open; let model shrink.

### D.1.3 Learned Ramp Size

**Goal.** Learn $R$ per-head via a second logit.

**Result.** High variance; unstable.

**Conclusion.** Keep $R$ as hyperparameter.

### D.1.4 Per-Layer Span Logits

**Goal.** Use different logits per (layer, head) instead of per-head shared.

**Result.** Similar accuracy, more parameters.

**Conclusion.** Per-head is sufficient.

### D.1.5 Content-Aware Span

**Goal.** Modulate span by content (query-specific).

**Result.** No improvement, 2× slower.

**Conclusion.** Distance-based span is sufficient.

## D.2 Abandoned Variants

### D.2.1 Exponential Ramp

Instead of linear ramp, use exponential decay. No improvement.

### D.2.2 Two-Sided Ramp

Ramp on both sides of span. No improvement.

### D.2.3 Multi-Span (Multiple Windows)

Multiple disjoint windows per head. No improvement.

## D.3 Failed Hyperparameters

### Ramp Size Too Small
| $R$ | DistilBERT | Note |
|---|---|---|
| 0.5 | 82.0 | Near hard |
| 1 | 87.0 | Sharp |
| 8 | 88.5 | OK |
| 32 | 89.5 | Best |
| 64 | 88.5 | Too smooth |

### L1 Too Aggressive
| $\lambda$ | DistilBERT |
|---|---|
| 0.1 | 86.0 |
| 0.5 | 78.0 |
| 1.0 | 65.0 |

Aggressive L1 collapses spans; accuracy drops.

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Hard truncation during training | Speed | Accuracy loss |
| Gradient clipping on span | Stability | No improvement |
| Warmup on span | Convergence | No improvement |
| Per-layer λ | Quality | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 89.5 |
| 123 | 89.3 |
| 456 | 89.7 |
| **Mean** | **89.5** |
| **Std** | **0.2** |

## D.6 Lessons Learned

1. **Start fully open, shrink via L1.**
2. **Soft ramp is essential** for differentiability.
3. **$R = 32$** is a good default.
4. **$\lambda = 0.01$** balances compression and accuracy.
5. **Learned spans are interpretable.**

---