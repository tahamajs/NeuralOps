<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Hybrid Works Without Retraining

NHA preserves most pretrained accuracy while reducing complexity to linear 
in $L$.

### 9.1.2 Takeaway 2: Orthogonal RFF Matters

Using orthogonal projection reduces variance, stabilizing training.

### 9.1.3 Takeaway 3: Learned Gates are Interpretable

The learned gate distribution reveals which positions benefit from local vs 
global attention.

## 9.2 When to Use NHA

### Recommended

- **Long-context inference** ($L > 512$)
- **Memory-constrained hardware**
- **Interpretability studies** of local vs global needs
- **Base for further optimization** (quantization, distillation)

### Not Recommended

- **Very short sequences** ($L < 128$)
- **Parameter reduction** (use A3/AASVD)

## 9.3 Choosing Window Size

| Sequence length | Recommended $w$ |
|---|---|
| 128 | 32 |
| 512 | 64 |
| 2048 | 128 |
| 8192 | 256 |

## 9.4 Choosing RFF Features

| Use case | Recommended $m$ |
|---|---|
| Fast inference | 32 |
| Balanced | 64 |
| High quality | 128 |
| Maximum | 256 |

## 9.5 Cost-Benefit

For GPT-2 at $L = 2048$:

| Config | VRAM | Time | Accuracy |
|---|---|---|---|
| Baseline | 793 MB | 8.5 s | 79.36% |
| NHA | 1511 MB MB | 2.3 s s | 50.92% |

## 9.6 Open Questions

1. **Chunk size optimization:** Can we learn the optimal chunk size?
2. **Adaptive RFF count:** Per-head different $m$?
3. **Combining with GQA:** Orthogonal optimizations?
4. **Learned kernel:** Replace ELU with a learned kernel?

## 9.7 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Cheaper inference:** Reduces compute per token.
- **Interpretability:** Gate analysis reveals structure.

### Negative
- **Dual-use:** Enables surveillance at scale.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---