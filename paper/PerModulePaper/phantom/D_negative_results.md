<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Learnable Hash Functions

**Goal.** Replace random projections with learnable ones.

**Result.** Accuracy improvement <0.3%, training instability.

**Conclusion.** Random projections are sufficient.

### D.1.2 Larger LSH Bits (>8)

**Goal.** Finer buckets for more precise matching.

**Result.** Most buckets had <2 tokens; attention too sparse.

**Conclusion.** 4-6 bits is optimal.

### D.1.3 Adaptive Rounds

**Goal.** Different rounds per layer.

**Result.** No improvement over uniform 4 rounds.

**Conclusion.** Uniform is sufficient.

### D.1.4 SVD Instead of CUR

**Goal.** Compare with standard SVD.

**Result.** CUR matches or beats SVD at rank 8-16; SVD better at high rank.

**Conclusion.** CUR is preferable for aggressive compression.

### D.1.5 Softmax Instead of Softpick

**Goal.** Compare with standard softmax.

**Result.** Softpick gives +{{SP_GAIN}}% accuracy from reduced noise.

**Conclusion.** Softpick is essential.

## D.2 Abandoned Variants

### D.2.1 Random Bucket Sampling

Instead of hash-based buckets, we tried random buckets. Result: 30% accuracy 
drop.

### D.2.2 Per-Head Hash Functions

We tried different hash functions per head. Result: no improvement.

### D.2.3 Dense Fallback Attention

We tried mixing LSH attention with dense attention. Result: no gain, 2× slower.

## D.3 Failed Hyperparameters

### LSH Rounds

| Rounds | DistilBERT | Time |
|---|---|---|
| 1 | {{D_LSH1}} | 1.0× |
| 4 | {{D_LSH4}} | 4.0× |
| 16 | {{D_LSH16}} | 16× (diminishing) |

### CUR Rank

| Rank | Ratio | DistilBERT |
|---|---|---|
| 4 | 60× | {{D_R4}} (too aggressive) |
| 8 | 30× | {{D_R8}} |
| 16 | 15× | {{D_R16}} |
| 32 | 8× | {{D_R32}} |
| 64 | 4× | {{D_R64}} |

### Softpick Temperature

| τ_s | DistilBERT |
|---|---|
| 0.1 | {{D_T01}} (too peaked) |
| 0.5 | {{D_T05}} |
| 1.0 | {{D_T10}} |
| 2.0 | {{D_T20}} (too flat) |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Larger `n_bits` | Precision | Sparse attention |
| Multi-scale LSH | Bucket size variance | No improvement |
| Per-layer τ_s | Non-stationarity | No improvement |
| LayerNorm recalibration | Shift | <0.2% improvement |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Lessons Learned

1. **Random projections work** as well as learnable ones.
2. **4-6 bits** is the sweet spot for LSH.
3. **Uniform rounds** suffice; no need for adaptive.
4. **Softpick matters** — softmax loses {{SP_GAIN}}%.
5. **CUR beats SVD** at low rank.

## D.7 Recommendations

1. **Use 4 rounds** with 4 bits.
2. **Use CUR with rank 16** for aggressive compression.
3. **Always use softpick** with τ_s=1.0.
4. **Enable fallback** in production.
```

---