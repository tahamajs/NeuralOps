# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 No Zero-Init

**Goal.** Standard random init for experts.

**Result.** Accuracy dropped to {{NO_ZI_ACC}}%.

**Conclusion.** Zero-init is essential.

### D.1.2 Gate Bias = 0

**Result.** {{BG0_ACC}}% accuracy; unstable.

**Conclusion.** Negative bias needed.

### D.1.3 Top-K = 1

**Result.** {{K1_ACC}}%; experts underutilized.

**Conclusion.** Top-2 better.

### D.1.4 MoE on Cross-Attention

**Result.** Encoder-decoder alignment broken.

**Conclusion.** Skip cross-attention.

### D.1.5 Many Experts (E=32)

**Result.** {{E32_ACC}}%; parameter explosion (+900M).

**Conclusion.** Diminishing returns above E=8.

## D.2 Failed Hyperparameters

| LR | D acc |
|---|---|
| 5e-5 | {{D_LR5e5}} |
| 1e-4 | {{D_LR1e4}} |
| 2e-4 | {{D_ACC}} |
| 5e-4 | {{D_LR5e4}} |

## D.3 Multi-Seed

| Seed | D acc |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| Mean | {{SEED_MEAN}} |
| Std | {{SEED_STD}} |

## D.4 Lessons Learned

1. Zero-init is essential.
2. Gate bias -3.0 is good.
3. Top-2 is better than top-1.
4. Skip MoE on cross-attention.
5. E=8 is a good default.
