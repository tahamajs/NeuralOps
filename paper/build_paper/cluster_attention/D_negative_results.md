# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Too Few Clusters (K=2)

**Result.** Accuracy dropped to 85.0%. Centroids too coarse.

### D.1.2 No Local Window

**Result.** 88.2%. Local structure lost.

### D.1.3 No Global Clustering

**Result.** 87.0%. Equivalent to local-only (Longformer-like).

### D.1.4 Hard Assignment (no temperature)

**Result.** 85.5%. Non-differentiable, unstable.

### D.1.5 Very Small Window (W=2)

**Result.** 86.5%. Local branch too narrow.

### D.1.6 10+ K-Means Iterations

**Result.** No improvement, 3× slower. 3 iterations suffice.

## D.2 Failed Hyperparameters

| LR | D acc |
|---|---|
| 5e-5 | 89.2 |
| 1e-4 | 89.8 |
| 2e-4 | 90.02 |
| 5e-4 | 88.5 (unstable) |

## D.3 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | 89.91 | 77.50 | 78.98 |
| 123 | 89.85 | 77.45 | 78.90 |
| 456 | 89.95 | 77.55 | 79.00 |
| Mean | 89.90 | 77.50 | 78.96 |
| Std | 0.05 | 0.05 | 0.05 |

## D.4 Lessons Learned

1. K=16 is a good default.
2. Local window essential.
3. Temperature clamp prevents collapse.
4. 3 iterations sufficient.
5. Causal decoders need exact attention (not clustering).

---