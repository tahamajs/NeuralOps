<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Symmetric Landmarks

**Goal.** Use same landmarks for Q and K (standard Nyström).

**Result.** Accuracy dropped by {{SYM_DROP}}% compared to asymmetric.

**Conclusion.** Asymmetry helps.

### D.1.2 Random Landmarks

**Goal.** Select m random positions as landmarks.

**Result.** Accuracy dropped by {{RANDOM_DROP}}%.

**Conclusion.** Segment pooling better.

### D.1.3 Learned Pooling

**Goal.** Add trainable projection for pooling weights.

**Result.** No improvement, 2× slower.

**Conclusion.** Mean pooling is sufficient.

### D.1.4 Schulz from Zero

**Goal.** Initialize $X_0 = 0$ or $X_0 = F_2^\top / \|F_2\|$.

**Result.** Often diverges or oscillates.

**Conclusion.** Initialize from SVD pinv.

### D.1.5 K-means Landmarks

**Goal.** Use k-means clustering to select landmarks.

**Result.** Accuracy similar to segment pooling, but 10× slower.

**Conclusion.** Segment pooling is a good trade-off.

## D.2 Abandoned Variants

### D.2.1 Per-Head Landmark Count

Different $m$ per head. No improvement.

### D.2.2 Adaptive $m$

Adjust $m$ based on sequence length. Marginal improvement.

### D.2.3 Multi-Scale Landmarks

Hierarchical pooling. No improvement.

## D.3 Failed Hyperparameters

### Too Few Landmarks
| $m$ | DistilBERT |
|---|---|
| 4 | {{D_M4}} (too low) |
| 8 | {{D_M8}} |
| 16 | {{D_M16}} |
| 32 | {{D_M32}} |

### Too Many Pinv Iterations
| $K$ | DistilBERT | Time |
|---|---|---|
| 4 | {{D_K4}} | 1.0× |
| 8 | {{D_K8}} | 1.2× |
| 16 | {{D_K16}} | 1.5× |

No improvement beyond $K = 4$.

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| FP16 pinv | Speed | NaN |
| Approx pinv | Speed | Accuracy loss |
| Fewer F factors | Speed | Accuracy loss |
| No RoPE before pooling | Speed | Accuracy loss |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Lessons Learned

1. **Asymmetric > symmetric** by {{ASYM_GAIN}}%.
2. **Segment pooling is sufficient.**
3. **Schulz from SVD pinv is essential.**
4. **$K = 4$ iterations suffice.**
5. **FP32 required for stability.**
