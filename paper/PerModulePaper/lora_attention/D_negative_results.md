# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 LoRA on FFN Only

**Goal.** LoRA on FFN (standard).

**Result.** Accuracy 86.4\%% vs LoRA Attention 89.91%.

**Conclusion.** Attention LoRA is more effective for classification.

### D.1.2 Q-Only LoRA

**Goal.** Reduce trainable params further.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** All projections matter.

### D.1.3 Very High Rank

**Goal.** $r_L = 256$.

**Result.** No improvement, 4× more params.

**Conclusion.** Rank 16 is sufficient.

### D.1.4 Very Low Rank

**Goal.** $r_L = 2$.

**Result.** Accuracy dropped significantly.

**Conclusion.** Rank 4 is the practical minimum.

### D.1.5 Non-Zero B Init

**Goal.** Random init for B.

**Result.** Initial loss much higher; slower convergence.

**Conclusion.** Zero-init B is essential.

## D.2 Abandoned Variants

### D.2.1 LoRA with Dropout on Adapters

Tested dropout on LoRA branch. No improvement.

### D.2.2 LoRA + Full FT on Classifier

Trained classifier head fully. Marginal improvement, defeats purpose.

### D.2.3 Different LR for A, B

Asymmetric LR (LoRA+ style). Small improvement, not worth complexity.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | 0.85 |
| 5e-5 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |

### Alpha

| alpha | DistilBERT |
|---|---|
| 8 | 0.85 |
| 16 | 0.85 |
| 32 | 0.85 |
| 64 | 0.85 |
| 128 | 0.85 |

## D.4 Lessons Learned

1. **Attention LoRA > FFN LoRA.**
2. **All four projections matter.**
3. **Zero-init B is essential.**
4. **Rank 16 is a good default.**
5. **LR 2e-4 works well.**

---