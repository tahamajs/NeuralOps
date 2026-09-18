# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Split-Head is Expressive

The head partition allows the model to learn which heads should focus on 
local patterns and which on global context. **Figure 9.1** shows the 
learned partition converges within 20 recovery steps.

### 9.1.2 Takeaway 2: Energy-Based SVD Requires Careful $r$

The default `max_rank=384` provides no compression. For real compression, 
lower to 128 or 192.

### 9.1.3 Takeaway 3: Softpick Fix Matters

The V4→V5 improvement on problematic layers is significant (see Section 7.9).

### 9.1.4 Takeaway 4: Decoder Fallback is Conservative

For decoders, we sacrifice complexity for correctness. Other methods (like 
LEKAv3) attempt to preserve both.

## 9.2 When to Use AEGIS-V5

### Recommended
- **Encoder-only models** (BERT-family)
- **Variable-length sequences** (adaptive window)
- **Combined with quantization** for INT8 deployment

### Not Recommended
- **Long-sequence decoders**: fallback negates benefit
- **Parameter-critical deployments**: check `max_rank`
- **Very short sequences** ($N < W$): fallback to standard

## 9.3 Decision Flowchart

```
Model is encoder? ─── Yes ─── Use AEGIS-V5 (with tuned max_rank)
  │ No
  ▼
Accept quadratic fallback? ─── Yes ─── Use AEGIS-V5
  │ No
  ▼
Consider LEKAv3 or other causal-efficient methods
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 128$:

| Metric | Baseline | AEGIS-V5 (r=384) | AEGIS-V5 (r=128) |
|---|---|---|---|
| Params | 66.95M | 0.85M | 0.85M |
| Accuracy | 90.71% | 86.4\% | 86.4\% |
| Inference time | 1.9 s | 0.85 s | 0.85 s |

## 9.5 Comparison to Prior AEGIS Versions

| Version | Split-head | Softpick | SVD | Multi-arch |
|---|---|---|---|---|
| V4 | ✗ | Broken | ✓ | ✓ |
| **V5** | **✓** | **Fixed** | **✓** | **✓** |
| V6 | ✓ | Fixed | ✓ | +fusion |

## 9.6 Open Questions

1. **Optimal head split:** Is 2:1 optimal? Or task-dependent?
2. **Adaptive rank:** Can SVD rank be learned per layer?
3. **Causal-efficient:** Can we avoid the decoder fallback?
4. **Interaction with quantization:** AEGIS-V5 + INT8?

## 9.7 Broader Impact

### Positive
- Enables efficient attention on consumer hardware
- Reduces compute for long-context encoders

### Negative
- Parameter count is unpredictable (depends on `max_rank`)
- Conservative decoder fallback limits benefit

---