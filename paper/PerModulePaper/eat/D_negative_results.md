<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Training with Pruning Enabled

**Goal.** Train EAT with pruning active.

**Result.** Gradient flow is broken through `argsort`. Loss diverged.

**Conclusion.** Pruning must be inference-only.

### D.1.2 EAT on RoBERTa with `is_nested_encoder = False`

**Goal.** Force EAT to apply its own output projection on RoBERTa.

**Result.** Double projection. Accuracy dropped by {{NESTED_DROP}}%.

**Conclusion.** `is_nested_encoder` must be `True` for RoBERTa.

### D.1.3 Without LoRA Recovery

**Goal.** Use EAT with only weight transfer, no recovery.

**Result.** Accuracy dropped by {{NORECOVERY_DROP}}% on RoBERTa.

**Conclusion.** LoRA recovery is essential.

### D.1.4 Aggressive Pruning (Ratio 0.3)

**Goal.** Prune 70% of tokens.

**Result.** Accuracy dropped by {{PRUNE30_DROP}}%.

**Conclusion.** Pruning ratio should not exceed 0.7.

### D.1.5 EAT without Frozen Base Weights

**Goal.** Let base weights be trainable too.

**Result.** 
- Training time increased 5×
- No accuracy improvement
- Parameter count much higher

**Conclusion.** LoRA alone is sufficient.

## D.2 Abandoned Variants

### D.2.1 Per-Layer LoRA Rank

**Goal.** Different LoRA rank per layer.

**Result.** No improvement over uniform rank, more hyperparameters.

### D.2.2 Adaptive Pruning Ratio

**Goal.** Learn pruning ratio during training.

**Result.** Unstable, no convergence.

### D.2.3 Merge Instead of Prune

**Goal.** Use token merging instead of pruning.

**Result.** Slower than pruning, similar accuracy.

## D.3 Failed Hyperparameters

### LoRA Rank

| Rank | DistilBERT | RoBERTa |
|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} |
| 64 | {{D_LR64}} | {{R_LR64}} |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | {{D_LR5e5}} |
| 2e-4 | {{D_LR2e4}} |
| 5e-4 | {{D_LR5e4}} |
| 1e-3 | {{D_LR1e3}} (NaN) |

## D.4 Attempted Fixes for Silent Failure

| Fix | Result |
|---|---|
| `logger.warning` on missing | No change (already present) |
| Raise on missing | Would break injection on some models |
| Fallback to unweighted | Not applicable (no compression) |
| Skip injection on missing | Model stays as baseline |

## D.5 Multi-Seed (Partial)

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| Mean | {{MEAN}} |
| Std | {{STD}} |

## D.6 Lessons Learned

1. **Pruning is inference-only.**
2. **`is_nested_encoder` must be correct.**
3. **LoRA recovery is essential.**
4. **Pruning ratio < 0.7 for accuracy.**
5. **Freeze base weights is best.**

---
