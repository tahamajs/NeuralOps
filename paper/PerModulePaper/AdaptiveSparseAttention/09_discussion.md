<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Per-Head Budgets Matter

Learned sparsity budgets are **highly non-uniform**. Using a single global 
budget (as static sparsity does) leaves capacity on the table for critical 
heads and wastes compute on redundant heads.

### 9.1.2 Takeaway 2: Soft Top-$k$ Works

The sigmoid-based soft mask is differentiable and effective. It avoids 
the complexity of perturbation or Gumbel relaxations.

### 9.1.3 Takeaway 3: Chunking is Essential

Without chunking, ASA's memory benefit is negligible. Chunking provides 
$\mathcal{O}(L \cdot C)$ peak memory regardless of query length.

## 9.2 When to Use ASA

### Recommended

- **Long-context inference** ($L > 1024$)
- **Memory-constrained hardware**
- **Interpretability studies** (head budget analysis)
- **Combined with GQA** for multiplicative savings

### Not Recommended

- **Very short sequences** ($L < 128$)
- **When parameter reduction is the goal** (use A3 or AASVD instead)
- **Applications requiring strict causality** (chunking complicates this)

## 9.3 Choosing Sparsity

| Use case | Recommended $\rho$ |
|---|---|
| Safe (minimal drop) | 0.3 |
| Balanced | 0.5 |
| Aggressive | 0.7 |
| Extreme | 0.9 |

## 9.4 Cost-Benefit

For GPT-2 at $L = 512$:

| Config | VRAM | Inference | Accuracy |
|---|---|---|---|
| Baseline | 793 MB | 4.2 s | 79.36% |
| ASA ($\rho=0.5$) | 550 MB | 4.6 s | 76.5% |

## 9.5 Open Questions

1. **Learned chunk size:** Can we learn the optimal chunk size?
2. **Cross-layer sparsity:** Should early and late layers have different 
   defaults?
3. **Combining with GQA:** What is the interaction?
4. **Theoretical bound:** How tight is our error bound?

## 9.6 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Cheaper inference:** Reduces compute per token.
- **Interpretability:** Head budgets reveal structure.

### Negative
- **Dual-use:** Enables surveillance at scale.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~10 GPU-hours, ~12.4 kg CO₂eq.

---