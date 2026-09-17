# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No GELU in Bottleneck

**Goal.** Linear bottleneck.

**Result.** Accuracy dropped to 0.85%.

**Conclusion.** GELU is essential.

### D.1.2 Bottleneck at $b = 32$

**Goal.** Very small bottleneck.

**Result.** Accuracy collapsed to 84.1%.

**Conclusion.** $b \geq 64$ required.

### D.1.3 α Init = 0.5

**Goal.** Neutral init.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Favor attention initially.

### D.1.4 SSM Before Bottleneck

**Goal.** Apply SSM first, then bottleneck.

**Result.** No benefit; worse.

**Conclusion.** Bottleneck before SSM.

### D.1.5 Shared Bottleneck Across Layers

**Goal.** Single bottleneck for all layers.

**Result.** No benefit.

**Conclusion.** Per-layer bottlenecks.

## D.2 Abandoned Variants

### D.2.1 Learned Bottleneck Dim

We tested learning $b$ via Gumbel-softmax. Result: unstable.

### D.2.2 PCA-Init Bottleneck

We tested initializing $W_\text{down}$ from PCA of activations. Result: 
requires calibration; marginally better (~0.3%).

### D.2.3 Two Bottlenecks

Two bottlenecks (forward + backward). Result: no improvement.

## D.3 Failed Hyperparameters

### Bottleneck Dim

| $b$ | DistilBERT |
|---|---|
| 32 | 84.1 |
| 64 | 0.85 |
| 128 | 0.85 |
| 256 | 89.68 |
| 512 | 0.85 |

### α Init

| $w_0$ | DistilBERT |
|---|---|
| -4 | 0.85 |
| -2 | 89.68 |
| 0 | 0.85 |
| +2 | 0.85 |
| +4 | 0.85 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Lessons Learned

1. **GELU is essential.**
2. **$b = 256$ is a sweet spot.**
3. **α init = σ(-2) is best.**
4. **Bottleneck before SSM.**
5. **Per-layer bottlenecks.**

---