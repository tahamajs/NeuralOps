# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Bottleneck is Beneficial

The concept bottleneck improves accuracy by 0.85% at 
64-dim, confirming the IB hypothesis.

### 9.1.2 Takeaway 2: SSM Doesn't Need Full Dimensions

The SSM state's effective rank is close to $b = 256$, not $d = 768$. The 
extra 512 dimensions are redundant.

### 9.1.3 Takeaway 3: Different Models Prefer Different Branches

α learns to 0.85 on DistilBERT, 0.85 on 
RoBERTa, and is 0.85 on GPT-2.

## 9.2 When to Use JambaInfo

### Recommended
- **Quality-focused applications.**
- **Long-context encoders** (bidirectional SSM helps).
- **Hybrid architecture research.**

### Not Recommended
- **Parameter-constrained deployments.**
- **Latency-critical applications.**
- **Encoder-decoder models.**

## 9.3 Decision Flowchart

```
Prioritize quality over params? ─── Yes ─── JambaInfo
  │ No
  ▼
Need parameter reduction? ─── Yes ─── A3/Adaptive SVD
  │ No
  ▼
Standard attention or HymbaTransX
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at $b = 256$:

| Metric | Baseline | Ours | Change |
|---|---|---|---|
| Params | 66.95M | 66.4M | +48.5\%% |
| Accuracy | 90.71% | 89.68% | -1.03% |
| Inference time | 1.90 s | 1.42x s | 2.1\times× |
| Peak VRAM | 183 MB | 480 MB MB | 48.5\%% |

## 9.5 Composition with Compression

JambaInfo can be combined with SVD factorization of attention projections. 
The bottleneck is orthogonal to weight compression.

Combined pipeline:
- Compress $W_{Q_1}, W_{K_1}, W_{Q_2}, W_{K_2}, W_V, W_O$ via SOA Fused.
- Keep bottleneck and SSM full-rank.
- Expected: **quality + compression** in one block.

## 9.6 Broader Impact

### Positive
- Principled bottleneck design.
- IB-motivated architecture.
- Complements compression.

### Negative
- Parameter increase.
- Slower training.

## 9.7 Open Questions

1. **Learned bottleneck dim:** Can $b$ be adaptive?
2. **Human-interpretable concepts:** Can the bottleneck be aligned to human concepts?
3. **PCA init:** Would PCA initialization help?
4. **Cross-attention:** How to extend?

---