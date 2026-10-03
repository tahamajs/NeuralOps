<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Formulation is Correct

The formulation of content-based sparse attention via k-means is sound. 
Accuracy is preserved within 1% on DistilBERT.

### 9.1.2 Takeaway 2: Implementation is Naive

The cluster loop is inefficient. Real speedups require sparse matmul kernels.

### 9.1.3 Takeaway 3: Not a Compression Method

Routing increases parameters via LoRA. It is not a compressor.

## 9.2 When to Use Routing

### 9.2.1 Recommended

- **Research on sparse attention**
- **Baseline for future efficient implementations**
- **Prototyping with small models**

### 9.2.2 Not Recommended

- **Production deployment** (slow)
- **Compression tasks**
- **Latency-critical applications**
- **Long sequences** without sparse kernels

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Need compression? ── Yes ── Use A3/SDRM
  │
  No
  ▼
Need speed? ── Yes ── Use FlashAttention/SDPA
  │
  No
  ▼
Want to study sparse attention? ── Yes ── Use Routing
  │
  No
  ▼
Consider alternatives
```

## 9.4 Comparison with Alternatives

| Method | Clustering | Speedup? | Complexity |
|---|---|---|---|
| Reformer | LSH | Sometimes | $\mathcal{O}(L \log L)$ |
| Routing Transformer | k-means | Sometimes | $\mathcal{O}(L \log L)$ |
| **Ours** | **k-means** | **No** | **$\mathcal{O}(K L^2)$** |

## 9.5 Open Questions

1. **Would sparse matmul help?** Requires custom kernels.
2. **Is Q+K better than Q?** Ablation shows minor improvement.
3. **Would per-query top-$k$ beat clustering?** Requires experiments.
4. **Can k-means be made differentiable?** Gumbel-Softmax is a candidate.

## 9.6 Broader Impact

### 9.6.1 Positive

- Contributes to sparse attention literature.
- Honest reporting of negative results.

### 9.6.2 Negative

- Not deployable as-is.

## 9.7 Recommendations for Production

If you must deploy routing:

1. **Implement sparse matmul kernels.**
2. **Precompute cluster assignments** for inference.
3. **Use Gumbel-Softmax** for differentiable clustering.
4. **Exclude padding** from k-means.

Without these, routing is not competitive with SDPA or FlashAttention.

---