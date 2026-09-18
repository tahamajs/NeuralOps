# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Max-Centering

**Goal.** Skip centering.

**Result.** NaN in FP16 from overflow.

**Conclusion.** Centering is essential.

### D.1.2 No ReLU Clamp

**Goal.** Skip ReLU.

**Result.** NaN from negative base.

**Conclusion.** ReLU is essential.

### D.1.3 Order 1

**Goal.** Linear polynomial.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Order 2 minimum.

### D.1.4 Order 16

**Goal.** High-order polynomial.

**Result.** No improvement, slower.

**Conclusion.** Order 4 is sufficient.

### D.1.5 Very High Temperature

**Goal.** $T = 10$.

**Result.** Attention became uniform.

**Conclusion.** $T$ should be near 1.

## D.2 Abandoned Variants

### D.2.1 Learned Order

Learned $d$ per layer. No benefit.

### D.2.2 Different Scaling

$s / \sqrt{d}$ instead of $s / d$. Unstable.

### D.2.3 No Dropout

Removed dropout. Slight overfitting.

## D.3 Failed Hyperparameters

### Order

| $d$ | DistilBERT |
|---|---|
| 1 | 0.85 |
| 2 | 0.85 |
| 4 | 0.85 |
| 8 | 0.85 |

### Temperature

| $T$ | DistilBERT |
|---|---|
| 0.5 | 0.85 |
| 1.0 | 0.85 |
| 2.0 | 0.85 |

## D.4 Lessons Learned

1. **Centering is essential.**
2. **ReLU clamp is essential.**
3. **Order 2 is minimum.**
4. **Order 4 is optimal.**
5. **Temperature near 1.**

---