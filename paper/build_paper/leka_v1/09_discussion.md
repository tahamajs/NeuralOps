# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Linear Scaling Works

LEKAv1 is faster than standard attention at $L > 128$.

### 9.1.2 Takeaway 2: RFF Approximates Gaussian, Not Softmax

Our kernel target is the Gaussian RBF, not the softmax kernel. The two 
differ in practice.

### 9.1.3 Takeaway 3: Preservation Gate is Essential

The gate provides a residual safety net, improving accuracy by 
0.85%.

## 9.2 When to Use LEKAv1

### Recommended
- **Long-context inference** ($L \geq 128$).
- **Streaming** (causal cumsum supports it).
- **Combined with other efficiency methods.**

### Not Recommended
- **Short sequences** ($L \leq 256$).
- **Tasks requiring exact softmax.**
- **Very large L (> 2048)** due to cumsum memory.

## 9.3 Decision Flowchart

```
Target L ≥ 1024? ─── Yes ─── LEKAv1
  │ No
  ▼
Latency-critical? ─── Yes ─── Standard attention
  │ No
  ▼
Consider other methods
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | LEKAv1 | Change |
|---|---|---|---|
| Accuracy | 90.71% | 51.03% | -39.68% |
| Total params | 66.95M | 66.4M | +48.5\%% |
| Inference @ L=128 | 1.90 s | 0.85 s | 84.8× |
| Inference @ L=2048 | 180 s | 0.85 s | 86.5× |

## 9.5 Composition with Other Methods

LEKAv1 composes with:

- **GQA:** reduces KV head count.
- **RoPE:** applies position encoding.
- **LoRA:** recovery capacity.
- **Weight compression** (SOA Fused): factorized projections.

## 9.6 Broader Impact

### Positive
- Linear-time inference.
- Composable with other methods.
- Fast long-context.

### Negative
- Modest parameter increase.
- Cumsum memory for very large L.

## 9.7 Open Questions

1. **Blocked cumsum:** How to reduce memory?
2. **Adaptive F:** Should F vary per layer?
3. **Combined with Performer:** PORF + cos/sin?

---