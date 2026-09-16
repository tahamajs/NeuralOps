<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Learned Spans are Interpretable

The learned spans reveal a **structural** pattern in the model: some heads 
are naturally local, others global. This is not visible from fixed-window 
methods.

### 9.1.2 Takeaway 2: Span Regularization Works

The L1 penalty causes spans to shrink gracefully, with minimal accuracy loss 
at $\lambda = {{LAMBDA}}$.

### 9.1.3 Takeaway 3: No Speedup Without Hard Truncation

The soft ramp is differentiable but **does not speed up inference** because 
all scores are still computed. To realize speedup, one would:
1. Train with soft ramp.
2. Convert to hard window at inference.
3. Use a sparse kernel.

## 9.2 When to Use ASA-Span

### Recommended

- **Interpretability studies** (head structure analysis)
- **Base for hard-window pruning**
- **Long-context fine-tuning**
- **Combined with other methods**

### Not Recommended

- **Immediate speedup needed** (use hard window directly)
- **Parameter-constrained** (use A3/AASVD)
- **Very short sequences** (no benefit)

## 9.3 Choosing Max Span

| Sequence length | Recommended $S_\text{max}$ |
|---|---|
| 128 | 64 |
| 512 | 128 |
| 2048 | 256 |
| 8192 | 512 |

**Rule of thumb.** $S_\text{max} = L / 4$ to $L / 2$.

## 9.4 Choosing Ramp Size

| Use case | Recommended $R$ |
|---|---|
| Aggressive pruning | 8 |
| Balanced | 16 |
| Smooth | 32 |
| Very smooth | 64 |

## 9.5 Cost-Benefit

For DistilBERT at $L = 128$:

| Config | Accuracy | Mean span | Reduction |
|---|---|---|---|
| Baseline | 90.71% | — | — |
| ASA-Span $\lambda=0$ | {{D_L0}}% | {{D_SPAN0}} | 0% |
| ASA-Span $\lambda=0.01$ | {{D_L01}}% | {{D_SPAN01}} | {{D_RED01}}% |
| ASA-Span $\lambda=0.1$ | {{D_L1}}% | {{D_SPAN1}} | {{D_RED1}}% |

## 9.6 Conversion to Hard Window

To realize speedup, one can:

1. **Train** with soft ramp.
2. **Extract** learned spans $\{z_h\}$.
3. **Round** to nearest power of 2 (for kernel alignment).
4. **Replace** soft ramp with hard band mask.
5. **Use sparse kernel** (e.g., block-sparse or flash attention with local window).

## 9.7 Open Questions

1. **Learned ramp size:** Can we learn $R$ per head?
2. **Interaction with content:** Does distance-based sparsity combine with 
   content-based (e.g., ASA-Sparse)?
3. **Cross-layer transfer:** Do spans transfer between tasks?
4. **Long-context scaling:** How do spans evolve at $L = 4096$?

## 9.8 Broader Impact

### Positive
- **Interpretability:** Reveals head structure.
- **Foundation for pruning:** Basis for hard-window deployment.
- **Efficiency:** Enables fixed-window deployment.

### Negative
- **Dual-use:** Enables surveillance at scale.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.
```

---