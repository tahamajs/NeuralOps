<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: LoRA is a Strong Distillation Target

Training only 1% of parameters (LoRA) recovers most of the accuracy lost 
by head-dim reduction.

### 9.1.2 Takeaway 2: Head-Dim Reduction Works

Reducing head dimension by 2× gives ~2× parameter 
reduction with acceptable loss.

### 9.1.3 Takeaway 3: Drop-In Wrapper

Squeezing Heads preserves pretrained weights exactly. It can be applied 
to any model without retraining.

## 9.2 When to Use Squeezing Heads

### Recommended

- **Model compression** where head pruning loses too much.
- **Continual learning** where you can't retrain from scratch.
- **Multi-task adaptation** — one LoRA per task.
- **Research on head structure** — study what's lost by dim reduction.

### Not Recommended

- **Inference speedup** — SH adds LoRA overhead.
- **Very small models** — LoRA overhead not worth it.
- **Training from scratch** — LoRA is a fine-tuning technique.

## 9.3 Choosing LoRA Rank

| Use case | Recommended $r_L$ |
|---|---|
| Extreme compression | 4 |
| Balanced | 8 |
| Safe | 16 |
| High quality | 32 |

## 9.4 Choosing Head-Dim Reduction

| Use case | Recommended $s$ |
|---|---|
| Safe | 1.0 (no reduction) |
| Balanced | 1.5 |
| Aggressive | 2.0 |
| Extreme | 2.5 |

## 9.5 Cost-Benefit

For DistilBERT at $L = 128$:

| Config | Trainable | Acc | Δ |
|---|---|---|---|
| Full FT | 66.9M | 90.71% | — |
| SH $r_L=16$ | 589824 | 90.25% | -0.46% |
| SH + $s=2$ | 589824 | 88.0% | -2.25% |

## 9.6 Open Questions

1. **Optimal per-head rank:** Can we allocate LoRA rank per head?
2. **Combining with pruning:** SH + head pruning?
3. **Distillation from teacher:** Can SH learn from a larger teacher?
4. **Cross-task transfer:** Does one LoRA transfer to another task?

## 9.7 Broader Impact

### Positive
- **Cheaper fine-tuning:** LoRA trains 1% of parameters.
- **Preserves pretrained:** No catastrophic forgetting.
- **Multi-task:** One adapter per task.

### Negative
- **Concentration:** Large labs benefit most.
- **Rebound effect:** Cheaper fine-tuning may increase total compute.

### Environmental
Total compute: ~10 GPU-hours, ~12.4 kg CO₂eq.

---
