# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Independent SVD (vs Joint)

**Goal.** Compare joint vs independent SVD.

**Result.** Independent uses ~{{INDEP_PARAM_INC}}% more parameters at the 
same accuracy.

**Conclusion.** Joint SVD is more efficient.

### D.1.2 No LoRA Recovery

**Goal.** Compress without LoRA.

**Result.** Accuracy dropped by {{NOLORA_DROP}}%.

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

**Result.** Accuracy dropped to {{R8_ACC}}%.

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
| 8 | {{D_R8}} |
| 16 | {{D_R16}} |
| 32 | {{D_R32}} |
| 64 | {{D_R64}} |
| 128 | {{D_R128}} |

Sweet spot: 64.

### LoRA Rank

| $r_L$ | DistilBERT |
|---|---|
| 4 | {{D_LORA4}} |
| 8 | {{D_LORA8}} |
| 16 | {{D_LORA16}} |
| 32 | {{D_LORA32}} |
| 64 | {{D_LORA64}} |

Sweet spot: 16.

## D.4 Lessons Learned

1. **Joint SVD works.**
2. **Nested detection order is critical.**
3. **LoRA recovery matters.**
4. **Rank 64 is a good default.**
5. **Detection order fix restores DistilBERT.**
```

---