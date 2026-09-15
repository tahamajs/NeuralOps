# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No GELU in Bottleneck

**Goal.** Linear bottleneck.

**Result.** Accuracy dropped to {{NO_GELU}}%.

**Conclusion.** GELU is essential.

### D.1.2 Bottleneck at $b = 32$

**Goal.** Very small bottleneck.

**Result.** Accuracy collapsed to {{B32}}%.

**Conclusion.** $b \geq 64$ required.

### D.1.3 α Init = 0.5

**Goal.** Neutral init.

**Result.** Accuracy dropped by {{ALPHA_HALF_DROP}}%.

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
| 32 | {{B32}} |
| 64 | {{D_B64}} |
| 128 | {{D_B128}} |
| 256 | {{D_ACC}} |
| 512 | {{D_B512}} |

### α Init

| $w_0$ | DistilBERT |
|---|---|
| -4 | {{D_WM4}} |
| -2 | {{D_ACC}} |
| 0 | {{D_W0}} |
| +2 | {{D_WP2}} |
| +4 | {{D_WP4}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.5 Lessons Learned

1. **GELU is essential.**
2. **$b = 256$ is a sweet spot.**
3. **α init = σ(-2) is best.**
4. **Bottleneck before SSM.**
5. **Per-layer bottlenecks.**

---
