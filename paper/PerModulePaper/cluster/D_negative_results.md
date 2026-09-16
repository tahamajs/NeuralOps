# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Too Few Clusters (K=2)

**Result.** Accuracy dropped to {{K2_ACC}}%. Centroids too coarse.

### D.1.2 No Local Window

**Result.** {{NO_LOCAL_ACC}}%. Local structure lost.

### D.1.3 No Global Clustering

**Result.** {{NO_GLOBAL_ACC}}%. Equivalent to local-only (Longformer-like).

### D.1.4 Hard Assignment (no temperature)

**Result.** {{HARD_ACC}}%. Non-differentiable, unstable.

### D.1.5 Very Small Window (W=2)

**Result.** {{W2_ACC}}%. Local branch too narrow.

### D.1.6 10+ K-Means Iterations

**Result.** No improvement, 3× slower. 3 iterations suffice.

## D.2 Failed Hyperparameters

| LR | D acc |
|---|---|
| 5e-5 | {{D_LR5e5}} |
| 1e-4 | {{D_LR1e4}} |
| 2e-4 | {{D_ACC}} |
| 5e-4 | {{D_LR5e4}} (unstable) |

## D.3 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | {{SEED42_D}} | {{SEED42_R}} | {{SEED42_G}} |
| 123 | {{SEED123_D}} | {{SEED123_R}} | {{SEED123_G}} |
| 456 | {{SEED456_D}} | {{SEED456_R}} | {{SEED456_G}} |
| Mean | {{MEAN_D}} | {{MEAN_R}} | {{MEAN_G}} |
| Std | {{STD_D}} | {{STD_R}} | {{STD_G}} |

## D.4 Lessons Learned

1. K=16 is a good default.
2. Local window essential.
3. Temperature clamp prevents collapse.
4. 3 iterations sufficient.
5. Causal decoders need exact attention (not clustering).

---