# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Truly O(1) Memory

DecayLinearAttention uses 16 KB of state memory per layer, independent of 
sequence length. This is a strict improvement over CHELA-SLAY and other 
cumsum-based methods.

### 9.1.2 Takeaway 2: Data-Dependent Decay Works

The learned per-token decay $\lambda_{h,t}$ improves accuracy by 
{{DECAY_GAIN}}% over fixed decay.

### 9.1.3 Takeaway 3: Cross-Attention Support

The same recurrence works for cross-attention, enabling encoder-decoder 
models.

## 9.2 When to Use DecayLinearAttention

### Recommended
- **Long-context inference** ($L \geq {{BREAKEVEN}}$).
- **Memory-constrained environments.**
- **Encoder-decoder models** (cross-attention).
- **Streaming decoding.**

### Not Recommended
- **Very short sequences** ($L \leq 256$).
- **When softmax is essential.**

## 9.3 Decision Flowchart

```
Need O(1) memory at long L? ─── Yes ─── Use DecayLinear
  │ No
  ▼
Need softmax behavior? ─── Yes ─── Use standard attention
  │ No
  ▼
Consider CHELA-SLAY (hybrid)
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at L = 2048:

| Metric | Baseline | DecayLinear | Change |
|---|---|---|---|
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Params | 66.95M | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| Time | 180 s | {{D_T2048}} s | {{D_RATIO_2048}}× |
| State mem | 12 MB | **16 KB** | **750×** |

## 9.5 Broader Impact

### Positive
- Enables long-context on consumer hardware.
- Reduces memory footprint dramatically.
- Cross-attention support broadens applicability.

### Negative
- Linear attention may not capture all softmax behaviors.
- Feature map choice affects accuracy.

## 9.6 Open Questions

1. **Learned feature map?** Would it improve over ELU+1?
2. **Per-head chunk sizes?** Would asymmetric $C$ help?
3. **Combined with token compression?**
4. **Scale to L > 32K?**

---