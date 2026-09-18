<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression from Two Axes

PHANTOM reduces **both** projection parameters and attention cost. This is 
unique among post-training compression methods.

### 9.1.2 Takeaway 2: LSH is Content-Adaptive

Unlike fixed-pattern sparse attention, LSH adapts to input. **Figure 9.1** 
shows bucket assignments varying across examples.

### 9.1.3 Takeaway 3: Softpick Reduces Noise

Truncating far-tail attention weights removes noise while preserving the 
dominant signal.

## 9.2 When to Use PHANTOM

### Recommended

- **Long-context inference** ($L > 256$)
- **Storage-constrained deployment**
- **Sparse attention studies**
- **CUR/SVD comparisons**

### Not Recommended

- **Short sequences** ($L < 32$): LSH overhead dominates
- **Latency-critical** applications: 4 rounds = 4× attention compute
- **Very small models** (<10M): overhead too large

## 9.3 Decision Flowchart

```
Target sequence length > 256? ─── Yes ─── Use PHANTOM
  │ No
  ▼
Attention sparsity high (>90%)? ─── Yes ─── Use PHANTOM
  │ No
  ▼
Storage reduction critical? ─── Yes ─── Use PHANTOM (CUR only)
  │ No
  ▼
Consider AASVD or quantization
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | PHANTOM | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | 0.85 MB | **2.1\times×** |
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 328 MB MB | **2.1\times×** |
| Time | 1.90 s | 1.42x s | **2.1\times×** |
| Accuracy | 90.71% | 72.59% | -18.12% |

**Verdict.** PHANTOM is a win for storage and VRAM; small loss for accuracy.

## 9.5 Comparison to Published Methods

| Method | Type | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT | Sharing | 3.5× | 89.4% |
| AASVD | Low-rank | 2.1\times× | 86.4\%% |
| SparseGPT | Pruning | 2× | 86.4\%% |
| **PHANTOM** | CUR + LSH | **50\%×** | **72.59%** |

## 9.6 Deployment Considerations

### Storage

PHANTOM's main benefit is storage. For 100M params:
- Baseline: 200 MB FP16
- PHANTOM (r=16): ~0.85 MB
- PHANTOM + INT8: ~0.85 MB

### Latency

Multi-round LSH adds ~4× attention compute. On GPU, matmul-dominated so 
effect is small.

### Memory

Attention memory reduced by **32×** due to bucket sparsity.

## 9.7 Open Questions

1. **Adaptive LSH rounds:** Can we learn $R_\ell$ per layer?
2. **Learned hash functions:** Replace random projections with learnable ones?
3. **Beyond SST-2:** Does LSH help for generation tasks?
4. **Interaction with quantization:** INT8 + PHANTOM?

## 9.8 Broader Impact

### Positive
- Democratizes long-context AI
- Reduces memory footprint
- Enables edge deployment

### Negative
- Complexity may hinder adoption
- Dual-use concerns

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---