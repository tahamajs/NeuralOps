# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 λ = 0.8 (Original Diff Transformer)

**Goal.** Use Ye et al.'s damping coefficient.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** λ = 0.2 is better for hybrid.

### D.1.2 No Normalization After ReLU

**Goal.** Skip the normalization step.

**Result.** Training diverged.

**Conclusion.** Normalization is essential.

### D.1.3 α Fixed at 0.5

**Goal.** No learned gate.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Learning α helps.

### D.1.4 No ReLU

**Goal.** Use signed differential attention.

**Result.** Unstable training, negative losses.

**Conclusion.** ReLU is essential.

### D.1.5 Shared Q Between Branches

**Goal.** Use the same Q for both differential branches.

**Result.** Degenerated to standard attention.

**Conclusion.** Independent Q is essential.

## D.2 Abandoned Variants

### D.2.1 Per-Head α

We tested per-head gates. Result: marginal benefit, more complexity.

### D.2.2 Multi-SSM Branches

Multiple SSM branches with independent decay. Result: worse than single.

### D.2.3 Deeper Differential

Three or more Q-K pairs. Result: no improvement.

## D.3 Failed Hyperparameters

### α Initialization

| Init | DistilBERT |
|---|---|
| σ(-2) ≈ 0.12 | 0.85 |
| σ(-1) ≈ 0.27 | 49.08 |
| σ(0) = 0.5 | 0.85 |
| σ(1) ≈ 0.73 | 0.85 |

### λ Sweep

| λ | DistilBERT |
|---|---|
| 0.0 | 0.0 |
| 0.2 | 82.4 |
| 0.5 | 85.1 |
| 0.8 | 86.0 |
| 1.0 | 86.4 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Lessons Learned

1. **λ = 0.2 (soft damping).**
2. **Normalization after ReLU is essential.**
3. **Learned α helps.**
4. **Independent Q for each branch.**
5. **ReLU prevents sign issues.**

---