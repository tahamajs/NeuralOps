<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 $G = 1$ (MQA)

**Goal.** Maximize KV reduction.

**Result.** Accuracy dropped to 78.0% (12.7%).

**Conclusion.** MQA too aggressive for these models.

### D.1.2 Random Group Assignment

**Goal.** Group random heads instead of consecutive.

**Result.** Accuracy dropped by 5%.

**Conclusion.** Head order matters; consecutive grouping is better.

### D.1.3 First-Head Only (No Averaging)

**Goal.** Pick the first head in each group.

**Result.** Accuracy dropped by 3%.

**Conclusion.** Averaging is better.

### D.1.4 Learned Pooling

**Goal.** Train a linear layer to combine heads.

**Result.** No improvement over averaging, 3× slower.

**Conclusion.** Averaging is sufficient.

## D.2 Abandoned Variants

### D.2.1 Non-Uniform Grouping

We tested grouping heads by similarity. No improvement over uniform.

### D.2.2 GQA with Different $G$ per Layer

We tested different $G$ for early vs late layers. No clear pattern.

### D.2.3 GQA + AASVD Together

Combining GQA with AASVD compression. Result: multiplicative parameter 
reduction but quality dropped significantly.

## D.3 Failed Hyperparameters

### LoRA Rank Too Small

| Rank | DistilBERT | Loss |
|---|---|---|
| 1 | 80.0 | Unstable |
| 2 | 85.0 | Stable |
| 4 | 86.5 | Stable |

### Learning Rate Too High

| LR | DistilBERT | Stability |
|---|---|---|
| 1e-3 | 50.92 | NaN |
| 5e-4 | 65.0 | Unstable |
| 2e-4 | 88.5 | Stable |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Group by similarity | Quality | No change |
| Learn pooling | Quality | No change |
| Per-layer $G$ | Quality | No pattern |
| Non-uniform group size | Quality | Worse |

## D.5 Long-Context Test (Not Fully Done)

We attempted $L = 4096$ but ran out of memory on RTX 4070 for some 
configurations. Partial results suggest GQA's benefit grows with $L$.

## D.6 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 88.5 |
| 123 | 88.3 |
| 456 | 88.7 |
| **Mean** | **88.5** |
| **Std** | **0.2** |

## D.7 Lessons Learned

1. **Averaging is the right choice** for grouping.
2. **Consecutive grouping works** without loss.
3. **$G = H/2$ is a safe default.**
4. **$G = H/4$ is aggressive but viable.**
5. **$G = 1$ (MQA) is too aggressive.**
