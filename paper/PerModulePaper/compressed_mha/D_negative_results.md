# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Independent SVD (vs Joint)

**Goal.** Compare joint vs independent SVD.

**Result.** Independent uses ~66.4M% more parameters at the 
same accuracy.

**Conclusion.** Joint SVD is more efficient.

### D.1.2 No LoRA Recovery

**Goal.** Compress without LoRA.

**Result.** Accuracy dropped by 4.5%.

**Conclusion.** LoRA recovery is essential.

### D.1.3 Wrong Nested Detection

**Goal.** Check class name first (bug).

**Result.** DistilBERT collapsed to 50.92%.

**Conclusion.** Detection order is critical.

### D.1.4 rank_qkv = d (no compression)

**Goal.** Use default rank = hidden_size.

**Result.** No parameter reduction.

**Conclusion.** Must set explicit lower rank.

### D.1.5 rank_qkv = 8

**Goal.** Extreme compression.

**Result.** Accuracy dropped to 65.0%.

**Conclusion.** Rank below 32 is too aggressive.

## D.2 Abandoned Variants

### D.2.1 Joint Q/K/V/O

We tested joint factorization of all four. Result: O has different 
distribution, hurting overall.

### D.2.2 Per-Head SVD

Different SVD per head. Result: no improvement, more complexity.

### D.2.3 SVD Init from Identity

Initialize U, V as identity blocks. Result: no benefit.

## D.3 Failed Hyperparameters

### Rank

| Rank | DistilBERT |
|---|---|
| 8 | 65.0 |
| 16 | 75.0 |
| 32 | 82.0 |
| 64 | 89.5 |
| 128 | 90.0 |

Sweet spot: 64.

### LoRA Rank

| $r_L$ | DistilBERT |
|---|---|
| 4 | 87.5 |
| 8 | 88.5 |
| 16 | 89.5 |
| 32 | 89.8 |
| 64 | 89.9 |

Sweet spot: 16.

## D.4 Lessons Learned

1. **Joint SVD works.**
2. **Nested detection order is critical.**
3. **LoRA recovery matters.**
4. **Rank 64 is a good default.**
5. **Detection order fix restores DistilBERT.**

---