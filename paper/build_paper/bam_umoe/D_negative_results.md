# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 No Zero-Init

**Goal.** Standard random init for experts.

**Result.** Accuracy dropped to 72.0%.

**Conclusion.** Zero-init is essential.

### D.1.2 Gate Bias = 0

**Result.** 84.5% accuracy; unstable.

**Conclusion.** Negative bias needed.

### D.1.3 Top-K = 1

**Result.** 88.5%; experts underutilized.

**Conclusion.** Top-2 better.

### D.1.4 MoE on Cross-Attention

**Result.** Encoder-decoder alignment broken.

**Conclusion.** Skip cross-attention.

### D.1.5 Many Experts (E=32)

**Result.** 89.6%; parameter explosion (+900M).

**Conclusion.** Diminishing returns above E=8.

## D.2 Failed Hyperparameters

| LR | D acc |
|---|---|
| 5e-5 | 88.5 |
| 1e-4 | 89.2 |
| 2e-4 | 90.02 |
| 5e-4 | 88.0 |

## D.3 Multi-Seed

| Seed | D acc |
|---|---|
| 42 | 89.5 |
| 123 | 89.4 |
| 456 | 89.6 |
| Mean | 89.5 |
| Std | 0.10 |

## D.4 Lessons Learned

1. Zero-init is essential.
2. Gate bias -3.0 is good.
3. Top-2 is better than top-1.
4. Skip MoE on cross-attention.
5. E=8 is a good default.

---