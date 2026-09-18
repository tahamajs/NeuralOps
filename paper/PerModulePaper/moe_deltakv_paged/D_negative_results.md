# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Aux Loss

**Goal.** Remove load balancing.

**Result.** Router collapsed to page 0 only.

**Conclusion.** Aux loss is essential.

### D.1.2 No CLS Boost

**Goal.** Remove page-0 boost.

**Result.** Accuracy dropped significantly (CLS not always selected).

**Conclusion.** Boost is essential.

### D.1.3 Very Small k_experts

**Goal.** $k = 1$.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** $k \geq 2$ recommended.

### D.1.4 Very Large Page Size

**Goal.** $P = 256$.

**Result.** No speedup (only 1-2 pages).

**Conclusion.** $P$ should be ~64.

### D.1.5 True Linear Scan

**Goal.** Replace cumsum with scan.

**Result.** No improvement, more complex.

**Conclusion.** Cumsum is fine.

## D.2 Abandoned Variants

### D.2.1 Adaptive Page Size

Different page sizes per layer. No improvement.

### D.2.2 Router Dropout

Dropout on router logits. Instability.

### D.2.3 Gradient on Delta

Added gradient on delta reconstruction. No benefit.

## D.3 Failed Hyperparameters

### SVD Rank

| Rank | DistilBERT |
|---|---|
| 4 | 32 |
| 8 | 32 |
| 16 | 32 |
| 32 | 32 |

Rank 16 is sweet spot.

### Aux Loss Weight

| λ | DistilBERT |
|---|---|
| 0.0 | 0.85 |
| 0.001 | 0.85 |
| 0.01 | 0.85 |
| 0.1 | 0.85 |

0.01 works well.

## D.4 Lessons Learned

1. **Aux loss is essential.**
2. **CLS boost is essential.**
3. **$k \geq 2$ needed.**
4. **$P = 64$ is good.**
5. **SVD rank 16 is good.**

---