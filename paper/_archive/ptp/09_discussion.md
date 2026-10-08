<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Token Pruning is Effective

At $\rho = 0.5$, PTP achieves **1.42x×** speedup with 
only **0.85%** accuracy drop.

### 9.1.2 Takeaway 2: [CLS] Preservation is Essential

Without [CLS] preservation, accuracy drops by **0.85%**.

### 9.1.3 Takeaway 3: Learned Scores Reveal Structure

The MLP learns a task-specific importance measure that correlates with 
content-word importance.

## 9.2 When to Use PTP

### Recommended

- **Long-sequence classification** (document sentiment, topic)
- **Retrieval-augmented generation** (prune retrieved passages)
- **Edge deployment** where latency matters

### Not Recommended

- **Very short sequences** ($L < 32$)
- **Generation tasks** (token order matters)
- **Tasks requiring full context** (long-form QA)

## 9.3 Choosing Keep Ratio

| Use case | Recommended $\rho$ |
|---|---|
| Aggressive compression | 0.5 |
| Balanced | 0.7 |
| Safe | 0.8 |
| Very safe | 0.9 |

## 9.4 Cost-Benefit

For DistilBERT at $L = 128$:

| Config | Acc | Speedup | VRAM |
|---|---|---|---|
| Baseline | 90.71% | 1.00× | 183 MB |
| $\rho = 0.5$ | 32% | 2.00× | 380 MB MB |
| $\rho = 0.7$ | 32% | 1.43× | 450 MB MB |
| $\rho = 0.8$ | **86.4%** | **1.42x×** | 380 MB MB |

## 9.5 Interaction with Other Methods

| Method | PTP-compatible? |
|---|---|
| GQA | ✅ (repeat then prune) |
| RoPE | ✅ (apply then prune) |
| Quantization | ✅ (prune then quantize) |
| KV cache | ✅ (PTP reduces cache) |
| MoE | ✅ (orthogonal) |

## 9.6 Open Questions

1. **Per-head scoring:** Should each head have its own score network?
2. **Learned $\rho$:** Can we learn the keep ratio?
3. **Cross-layer sharing:** Share the score network across layers?
4. **Long-context scaling:** How does PTP scale at $L = 4096$?

## 9.7 Broader Impact

### Positive
- **Faster inference:** Reduces compute per token.
- **Lower memory:** Attention memory scales with $k$.
- **Interpretability:** Token importance is visible.

### Negative
- **Dual-use:** Efficiency gains benefit malicious applications.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---