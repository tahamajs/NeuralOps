# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Dense Anchor is Essential

Pure linear attention loses ~28.0% accuracy. The dense anchor 
recovers this.

### 9.1.2 Takeaway 2: Spherical + Yat-Kernel Improves Linear

The combined spherical normalization + Yat-Kernel feature map gives a 
**stable, efficient** linear stream.

### 9.1.3 Takeaway 3: Cumsum Memory is the Limitation

The causal linear attention via cumsum materializes 
$(\text{B}, \text{H}, L, 2F, d_h)$ which is **memory-intensive** for short L.

## 9.2 When to Use CHELA-SLAY

### Recommended
- **Long-context inference** ($L \geq 1024$) where dense dominates.
- **Research on hybrid attention.**
- **Learning gate behavior** for stream selection.

### Not Recommended
- **Short sequences** ($L \leq 256$).
- **Memory-constrained environments** (cumsum dominates).
- **Latency-critical applications.**

## 9.3 Decision Flowchart

```
Target L ≥ 512? ─── Yes ─── Consider CHELA-SLAY
  │ No
  ▼
Need memory efficiency? ─── Yes ─── Use SOA Fused / Adaptive SVD
  │ No
  ▼
Consider standard attention
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at L = 128:

| Metric | Baseline | CHELA-SLAY | Change |
|---|---|---|---|
| Accuracy | 90.71% | 89.91% | -0.80% |
| Params | 66.95M | 67500000 | +0.8% |
| Time | 1.90 s | 4.2 s | 2.2× |
| VRAM | 183 MB | 374 MB MB | 4.9× |

## 9.5 Improving the Cumsum

**Future work.** Replace the cumsum with a **chunked state-passing**:

```python
for chunk_start in range(0, L, C):
    chunk_end = chunk_start + C
    # intra-chunk: cumsum within C tokens
    # inter-chunk: add previous chunk's final state
```

This would reduce memory to $\mathcal{O}(C \cdot 2F \cdot d_h)$.

## 9.6 Broader Impact

### Positive
- Contributes to hybrid attention research.
- Demonstrates stable linear attention.
- Provides multi-architecture support.

### Negative
- Cumsum memory limitation.
- Slower at short L.
- Modest accuracy gain.

## 9.7 Open Questions

1. **Chunked cumsum:** Would it eliminate the memory bottleneck?
2. **Learned Ω:** Could training $\Omega$ improve the linear stream?
3. **Different gate shapes:** Would a per-token per-dimension gate help?
4. **Scale to L > 4096:** Does the linear stream become dominant?

---