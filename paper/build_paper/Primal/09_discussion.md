<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Asymmetric Kernel Approximation

Allowing different landmarks for $Q$ and $K$ improves over symmetric Nyström 
by a small but consistent margin.

### 9.1.2 Takeaway 2: Length Advantage

Primal Attention's benefit grows with sequence length. At $L = 128$, 
inference is slower; at $L > 512$, it is faster.

### 9.1.3 Takeaway 3: No Parameter Reduction

Primal Attention does **not** reduce parameter count. For parameter 
reduction, use A3 or AASVD.

## 9.2 When to Use Primal Attention

### Recommended

- **Long sequences** ($L > 512$)
- **Memory-constrained inference** (attention dominates)
- **Combined with quantization** for extra savings

### Not Recommended

- **Short sequences** ($L < 128$)
- **Parameter-constrained** applications
- **Causality-critical** generation tasks

## 9.3 Choosing $m$

| $L$ | Recommended $m$ |
|---|---|
| < 128 | $L / 4$ |
| 128 – 512 | 64 |
| 512 – 2048 | 128 |
| > 2048 | 256 |

## 9.4 Cost-Benefit

For GPT-2 at $L = 2048$, $B = 1$:

| Metric | Baseline | Primal ($m=64$) |
|---|---|---|
| Attention memory | 12.6 MB | 0.20 MB (**64×**) |
| Inference time | 8.5 s | 6.5 s |
| Accuracy | 79.36% | 63.07% |

## 9.5 Open Questions

1. **Learned landmark selection:** Can we train the pooling?
2. **Content-aware pooling:** Use attention-weighted pooling?
3. **True causal approximation:** How to ensure causality?
4. **Combining with GQA:** Multiplicative savings?
5. **Theoretical error bound:** Tighter than Frobenius?

## 9.6 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Cheaper inference:** Reduces compute per token.
- **Edge deployment:** Makes on-device LLMs more feasible.

### Negative
- **Dual-use:** Enables long-context surveillance.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~12 GPU-hours, ~12.4 kg CO₂eq.

---
