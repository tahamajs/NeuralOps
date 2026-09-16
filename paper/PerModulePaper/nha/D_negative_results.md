<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Per-Head Gate

**Goal.** Replace token-level gate with per-head gate.

**Result.** No improvement, 3× more parameters.

**Conclusion.** Token-level is sufficient.

### D.1.2 i.i.d. Gaussian RFF

**Goal.** Use i.i.d. Gaussian instead of orthogonal.

**Result.** Accuracy dropped by {{IID_DROP}}%.

**Conclusion.** Orthogonal matters.

### D.1.3 ReLU Feature

**Goal.** Use ReLU instead of ELU.

**Result.** Not smooth; unstable gradients.

**Conclusion.** ELU is better.

### D.1.4 Gate Bias = 0

**Goal.** Start balanced.

**Result.** Model drifted toward global; accuracy dropped.

**Conclusion.** Start local-biased.

### D.1.5 Full Cumsum (No Chunking)

**Goal.** Compute global RFF with full cumsum.

**Result.** OOM for $L > 512$.

**Conclusion.** Chunked is necessary.

## D.2 Abandoned Variants

### D.2.1 Two-Level Fusion

Hierarchical gating. No improvement.

### D.2.2 Learned Window Size

Learn $w$ per head. No improvement, unstable.

### D.2.3 Learned RFF Count

Learn $m$ per head. No improvement.

## D.3 Failed Hyperparameters

### Window Size Too Small
| $w$ | DistilBERT |
|---|---|
| 4 | {{D_W4}} |
| 8 | {{D_W8}} |
| 16 | {{D_W16}} |
| 32 | {{D_W32}} |
| 64 | {{D_W64}} |

### RFF Count Too Small
| $m$ | DistilBERT |
|---|---|
| 16 | {{D_M16}} |
| 32 | {{D_M32}} |
| 64 | {{D_M64}} |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Gradient clip on gate | Stability | No improvement |
| Learned chunk size | Speed | No improvement |
| Per-layer window | Quality | No improvement |
| Separate projection for Q, K | Quality | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Lessons Learned

1. **Orthogonal RFF matters.**
2. **Token-level gate suffices.**
3. **Bias 3.0 at init is essential.**
4. **Chunked scan is necessary.**
5. **ELU > ReLU for stability.**

---