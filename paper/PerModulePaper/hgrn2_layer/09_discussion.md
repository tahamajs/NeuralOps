# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Element-Wise Recurrence Works

HGRN2Layer replaces attention with $\mathcal{O}(L d)$ element-wise recurrence, 
achieving accuracy within {{D_DELTA}}% of baseline on DistilBERT.

### 9.1.2 Takeaway 2: Short Sequences are Slow

At L = 128, our method is **{{SLOWDOWN}}× slower** than baseline due to Python 
chunk loops.

### 9.1.3 Takeaway 3: Parameters Increase

HGRN2Layer increases parameters by **{{D_PARAM_PCT}}%** per layer. We are 
honest about this.

## 9.2 When to Use HGRN2Layer

### Recommended
- **Long-context inference** ($L \geq {{BREAKEVEN}}$).
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
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Params | 66.95M | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| Time | 45 s | {{D_T1024}} s | {{S1024}}× |
| VRAM | 1500 MB | {{VRAM_1024}} MB | {{VRAM_1024_PCT}}% |

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
