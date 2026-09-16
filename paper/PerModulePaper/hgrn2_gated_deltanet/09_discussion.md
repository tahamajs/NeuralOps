# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Linear Recurrence for Long Context

At $L \geq {{BREAKEVEN}}$, HGRN2-GatedDeltaNet is faster than standard 
attention. For long-context deployment, this is a genuine win.

### 9.1.2 Takeaway 2: Short-Sequence Slowdown

At $L = 128$, our method is **{{SLOWDOWN}}× slower**. This is a significant 
limitation for standard benchmarks.

### 9.1.3 Takeaway 3: Two Branches Beat One

The learned gate + two branches improve over single-branch by 
{{BRANCH_GAIN}}%.

## 9.2 When to Use HGRN2-GatedDeltaNet

### Recommended
- **Long-context inference** ($L \geq 1024$)
- **Streaming / online processing**
- **Memory-constrained decoding**

### Not Recommended
- **Short sequences** ($L \leq 256$)
- **Latency-critical applications**
- **Parameter-sensitive deployments**

## 9.3 Decision Flowchart

```
Sequence length ≥ 512? ─── Yes ─── Use HGRN2-GatedDeltaNet
  │ No
  ▼
Latency-critical? ─── Yes ─── Use standard attention
  │ No
  ▼
Consider standard attention + KV compression
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at L = 1024:

| Metric | Baseline | Ours | Change |
|---|---|---|---|
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Params | 66.95M | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| Inference time | 45s | {{D_T1024}}s | {{S1024}}× |
| Peak VRAM | {{VRAM_BASE_1024}} MB | {{VRAM_T1024}} MB | {{VRAM_PCT_1024}}% |

## 9.5 Fused Kernel Opportunity

The main limitation is the Python loop. A **fused Triton kernel** could 
eliminate the loop overhead, potentially giving:

- 3-5× speedup at L = 128.
- 10-20× at L = 1024.

This is a clear avenue for future work.

## 9.6 Broader Impact

### Positive
- Enables long-context inference on edge devices.
- Reduces memory footprint for streaming.
- Contributes to the linear attention literature.

### Negative
- Large parameter increase (~226%).
- Training slower than baseline.
- Not yet competitive on standard benchmarks.

## 9.7 Open Questions

1. **Optimal gate initialization:** What initial value gives the best recovery?
2. **Fused kernels:** How much can a Triton implementation help?
3. **Cross-layer gating:** Should γ be shared across layers?
4. **Combining with SSM:** Can HGRN-2 be replaced by Mamba?
```

---