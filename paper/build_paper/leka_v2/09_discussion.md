# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Landmark Distillation Works

Learned landmark selection improves accuracy over random selection by 
0.85%.

### 9.1.2 Takeaway 2: Local-Global Fusion Matters

Combining local sliding window with global landmark attention improves 
accuracy by 0.85%.

### 9.1.3 Takeaway 3: Softpick Stabilizes Masked Attention

Softpick prevents uniform collapse on fully masked rows.

### 9.1.4 Takeaway 4: Not Truly Linear at ρ = 0.5

The current implementation is $O(L \cdot (W + \rho L))$, which is 
still quadratic. To be truly linear, set $\rho$ to a small constant 
ratio.

## 9.2 When to Use LEKAv2

### Recommended
- **Moderate-length sequences** (L ≤ 2048) with moderate ρ.
- **Tasks where some positions are much more important** (e.g., long 
  document classification with prominent sections).
- **When local context matters** as much as global.

### Not Recommended
- **Very long sequences** (L > 4096): the landmark term becomes expensive.
- **Latency-critical applications.**
- **Encoder-decoder models.**

## 9.3 Decision Flowchart

```
L ≤ 2048? ─── Yes ─── LEKAv2
  │ No
  ▼
Need truly linear? ─── Yes ─── LEKAv1 or Performer
  │ No
  ▼
Standard attention or Jamba-style hybrid
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at ρ = 0.25:

| Metric | Baseline | LEKAv2 | Change |
|---|---|---|---|
| Accuracy | 90.71% | 86.01% | -4.70% |
| Total params | 66.95M | 66.4M | +48.5\%% |
| Inference @ L=128 | 1.90 s | 0.85 s | 84.8× |
| Inference @ L=1024 | 45.0 s | 0.85 s | 86.4× |

## 9.5 Composition with Other Methods

LEKAv2 composes with:

- **GQA:** reduces KV head count.
- **RoPE:** position encoding.
- **LoRA:** recovery capacity.
- **Quantization:** INT8 on projection weights.

## 9.6 Broader Impact

### Positive
- Learnable landmark selection.
- Local-global fusion.
- Robust masked attention.

### Negative
- Not truly linear at default ρ.
- Modest parameter increase.

## 9.7 Open Questions

1. **Truly linear version:** Can we set $L_\text{max} = $ constant?
2. **Adaptive ρ:** Can ρ be learned?
3. **Cross-attention:** How to extend?
4. **Landmark transfer:** Do learned landmarks transfer across tasks?

---