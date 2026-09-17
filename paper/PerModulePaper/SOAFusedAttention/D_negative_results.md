# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Factorizing $W_O$

**Goal.** Also factorize the output projection.

**Result.** Accuracy dropped by 3.0%.

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

**Result.** Accuracy collapsed to 80.0%.

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
| 16 | 70.0 | Too small |
| 32 | 80.0 | Marginal |
| 64 | 86.0 | Good |
| 128 | 88.5 | Sweet spot |
| 384 | 90.5 | Near-lossless |

### LoRA Rank

| $r_L$ | DistilBERT |
|---|---|
| 0 (no LoRA) | 83.0 |
| 4 | 85.0 |
| 8 | 87.0 |
| 16 | 88.5 |
| 32 | 88.6 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 88.5 |
| 123 | 88.3 |
| 456 | 88.7 |
| **Mean** | **88.5** |
| **Std** | **0.2** |

## D.5 Lessons Learned

1. **Do not factorize $W_O$.**
2. **LoRA on up, not down.**
3. **Asymmetric split is better.**
4. **$k \geq 64$ required.**
5. **Frobenius check is essential.**

---