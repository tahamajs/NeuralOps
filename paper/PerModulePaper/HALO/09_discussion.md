<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Gates Reveal Head Structure

The learned gates show which heads benefit from dense attention and which 
can use linear. This is a useful **interpretability** signal.

### 9.1.2 Takeaway 2: Layered Hybrid Works

Early layers need global context; late layers can use linear. This 
validates architectural priors from prior work.

### 9.1.3 Takeaway 3: No Speedup Without Hard Routing

HALO-ReGLA computes both streams. To realize speedup:
1. Train with both streams.
2. Identify heads where $\bar{g}_h < \theta$.
3. Convert to pure linear inference.

Expected speedup: up to 3× if all heads convert.

## 9.2 When to Use HALO-ReGLA

### Recommended

- **Long-context inference** where some heads can be linearized
- **Interpretability studies** of head specialization
- **Base for hard routing** to linear-only inference

### Not Recommended

- **Short sequences** (no benefit)
- **Parameter reduction** (use A3/AASVD)
- **Immediate speedup** (use pure linear or sparse)

## 9.3 Choosing Gate Bias

| Bias | Behavior at init | Use case |
|---|---|---|
| 0 | Balanced | Uncertain |
| 1.0 | Dense-leaning | Safe default |
| 1.8 | Mostly dense | **Pretrained recovery** |
| 3.0 | Almost dense | Preserve baseline |

## 9.4 Cost-Benefit

For DistilBERT at $L = 128$:

| Config | Acc | VRAM | Time | Mean gate |
|---|---|---|---|---|
| Baseline | 90.71% | 183 MB | 1.9 s | — |
| HALO | 89.5% | 210 MB | 2.8 s | 0.72 |

## 9.5 Open Questions

1. **Learned threshold for hard routing:** Can we learn $\theta$?
2. **Interaction with KV cache:** Does HALO-ReGLA reduce KV cache?
3. **Long-context scaling:** How do gates evolve at $L = 4096$?
4. **Cross-task transfer:** Do gates transfer between tasks?

## 9.6 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Interpretability:** Reveals head structure.
- **Foundation for routing:** Basis for efficient deployment.

### Negative
- **Dual-use:** Enables surveillance at scale.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~15 GPU-hours, ~12.4 kg CO₂eq.

---
