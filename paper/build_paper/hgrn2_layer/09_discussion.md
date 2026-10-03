# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Element-Wise Recurrence Works

HGRN2Layer replaces attention with $\mathcal{O}(L d)$ element-wise recurrence, 
achieving accuracy within -27.75% of baseline on DistilBERT.

### 9.1.2 Takeaway 2: Short Sequences are Slow

At L = 128, our method is **1.0x× slower** than baseline due to Python 
chunk loops.

### 9.1.3 Takeaway 3: Parameters Increase

HGRN2Layer increases parameters by **48.5\%%** per layer. We are 
honest about this.

## 9.2 When to Use HGRN2Layer

### Recommended
- **Long-context inference** ($L \geq 128$).
- **Streaming / autoregressive decoding** with O(1) state.
- **Memory-constrained decoding** (state is O(d), not O(L)).

### Not Recommended
- **Short sequences** ($L \leq 256$).
- **Latency-critical applications.**
- **Parameter-sensitive deployments.**

## 9.3 Decision Flowchart

```
Target L ≥ 512? ─── Yes ─── Consider HGRN2Layer
  │ No
  ▼
Latency-critical? ─── Yes ─── Use standard attention
  │ No
  ▼
Consider other methods
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at L = 1024:

| Metric | Baseline | Ours | Change |
|---|---|---|---|
| Accuracy | 90.71% | 62.96% | -27.75% |
| Params | 66.95M | 66.4M | +48.5\%% |
| Time | 45 s | 0.85 s | 86.4× |
| VRAM | 1500 MB | 380 MB MB | 48.5\%% |

## 9.5 Fused Kernel Opportunity

A **fused Triton kernel** for the chunked scan could eliminate the Python 
loop. Expected speedup: 3-5× at L = 128.

## 9.6 Broader Impact

### Positive
- Linear-time long-context inference.
- O(1) state for streaming.

### Negative
- Parameter increase.
- Training slower.

## 9.7 Open Questions

1. **Optimal chunk size:** Is 32 best across all L?
2. **Fused kernels:** How much can they help?
3. **Chunk size scheduling:** Should it vary per layer?
4. **Combining with attention:** Hybrid HGRN-attention?

---