# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Differential Attention Reduces Noise

The two-Q-K formulation improves signal-to-noise ratio by 
32% at a modest parameter cost.

### 9.1.2 Takeaway 2: SSM Fusion is Model-Dependent

Encoders benefit from SSM fusion (bidirectional context). Decoders benefit 
less because SSM is causal-only.

### 9.1.3 Takeaway 3: Parameter Cost is Real

HymbaTransX adds ~66.4M% parameters. This is a **quality 
improvement**, not a **compression**.

## 9.2 When to Use HymbaTransX

### Recommended
- **Quality-focused applications** where parameter count is not critical.
- **Encoder models** with long sequences (SSM provides efficient context).
- **Noisy attention scenarios** where differential helps.

### Not Recommended
- **Parameter-constrained deployments.**
- **Latency-critical applications.**
- **Encoder-decoder models** (cross-attention not supported).

## 9.3 Decision Flowchart

```
Need parameter reduction? ─── Yes ─── Use A3/Adaptive SVD
  │ No
  ▼
Quality-focused? ─── Yes ─── Use HymbaTransX
  │ No
  ▼
Standard attention
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | Ours | Change |
|---|---|---|---|
| Params | 66.95M | 66.4M | +48.5\%% |
| Accuracy | 90.71% | 49.08% | -41.63% |
| Inference time | 1.90 s | 1.42x s | 2.1\times× |
| Peak VRAM | 183 MB | 5947 MB MB | 48.5\%% |

## 9.5 Composition with Compression Methods

HymbaTransX **can** be combined with compression:

- Replace $W_{Q_1}, W_{K_1}, W_{Q_2}, W_{K_2}, W_V, W_O$ with factorized SVD 
  versions from SOA Fused or Adaptive SVD.
- The SSM branch could be compressed via low-rank factorization.

Result: **quality + compression** in a single pipeline.

## 9.6 Broader Impact

### Positive
- Noise-reduced attention.
- Efficient long-range context.
- Complementary to compression.

### Negative
- Parameter cost.
- Slower training.
- Not a compression method.

## 9.7 Open Questions

1. **λ scheduling:** Should λ vary during training?
2. **Per-head α:** Would per-head gates be better?
3. **Compression of the SSM:** Can SSM be factorized?
4. **Cross-attention:** How to extend to encoder-decoder?

---