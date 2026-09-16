# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Factorizing $W_O$

**Goal.** Also factorize the output projection.

**Result.** Accuracy dropped by {{WO_FACTOR_DROP}}%.

**Conclusion.** Keep $W_O$ intact.

### D.1.2 LoRA on Down-Projection

**Goal.** Attach LoRA to the down-projection.

**Result.** No improvement; slightly worse.

**Conclusion.** LoRA on up is better.

### D.1.3 Symmetric Split with $\Sigma^{1/2}$

**Goal.** Use symmetric split.

**Result.** Numerically less stable in FP16.

**Conclusion.** Asymmetric split (all $\Sigma$ in up) is better.

### D.1.4 Very Low Rank ($k < 32$)

**Goal.** Extreme compression.

**Result.** Accuracy collapsed to {{K32_ACC}}%.

**Conclusion.** $k \geq 64$ required.

### D.1.5 No Frobenius Check

**Goal.** Skip the validation.

**Result.** Silent failures on some models.

**Conclusion.** Keep the check.

## D.2 Abandoned Variants

### D.2.1 Per-Head Factorization

We tested factorizing each head's projection separately. Result: no 
improvement, more complexity.

### D.2.2 Joint QKV Factorization

We tested concatenating Q, K, V into one matrix and factorizing jointly. 
Result: worse reconstruction because the joint matrix has different spectral 
structure.

### D.2.3 Rank per Layer

We tested different ranks per layer. Result: marginal benefit, needs tuning.

## D.3 Failed Hyperparameters

### Rank

| $k$ | DistilBERT | Note |
|---|---|---|
| 16 | {{D_R16_ACC}} | Too small |
| 32 | {{D_R32}} | Marginal |
| 64 | {{D_R64}} | Good |
| 128 | {{D_R128}} | Sweet spot |
| 384 | {{D_R384}} | Near-lossless |

### LoRA Rank

| $r_L$ | DistilBERT |
|---|---|
| 0 (no LoRA) | {{D_LR0}} |
| 4 | {{D_LR4}} |
| 8 | {{D_LR8}} |
| 16 | {{D_LR16}} |
| 32 | {{D_LR32}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.5 Lessons Learned

1. **Do not factorize $W_O$.**
2. **LoRA on up, not down.**
3. **Asymmetric split is better.**
4. **$k \geq 64$ required.**
5. **Frobenius check is essential.**
```

---