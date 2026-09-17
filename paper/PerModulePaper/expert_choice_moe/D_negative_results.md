# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Zero-Init

**Goal.** Skip zero-init of expert output.

**Result.** Loss diverged.

**Conclusion.** Zero-init is essential.

### D.1.2 Capacity Factor < 0.5

**Goal.** Very aggressive compression.

**Result.** Underutilization; accuracy degraded.

**Conclusion.** $CF \geq 0.5$ required.

### D.1.3 MoE on Values (like BAM-UMoE)

**Goal.** Apply MoE to V instead of attention output.

**Result.** Worse than attention output.

**Conclusion.** Attention output is the right target.

### D.1.4 Token-Choice in This Framework

**Goal.** Use token-choice routing.

**Result.** Expert imbalance; needed aux loss.

**Conclusion.** Expert-choice is simpler.

### D.1.5 Top-1 Per Expert

**Goal.** Each expert picks 1 token.

**Result.** Underutilization; too few updates.

**Conclusion.** Larger $k$ needed.

## D.2 Abandoned Variants

### D.2.1 Shared Experts

We tested sharing expert parameters across layers. Result: much worse 
accuracy.

### D.2.2 Learned Capacity Factor

We tested making CF learnable per expert. Result: complex; marginal 
benefit.

### D.2.3 Attention-Only MoE (No Gate)

We tested MoE without the gate. Result: unstable training.

## D.3 Failed Hyperparameters

### Expert Count

| E | DistilBERT | Note |
|---|---|---|
| 2 | 0.85 | Too few |
| 4 | 0.85 | Marginal |
| 8 | 0.85 | Good |
| 16 | 0.85 | Marginal gain, 2× params |
| 32 | OOM | — |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Lessons Learned

1. **Zero-init is essential.**
2. **Expert-choice > token-choice** for load balance.
3. **Attention output > values** for MoE placement.
4. **CF ≥ 0.5 required.**
5. **8 experts is a good default.**

---