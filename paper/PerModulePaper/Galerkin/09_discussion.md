# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Encoder-Only Benefit

Galerkin provides linear complexity for **encoder** models. Decoders use 
fallback.

### 9.1.2 Takeaway 2: DCT is the Right Basis

The smoothness of attention operators makes DCT the natural basis. Random 
orthonormal bases underperform.

### 9.1.3 Takeaway 3: Not a Compression Method

Parameter count increases via LoRA. Not for size reduction.

## 9.2 When to Use Galerkin Attention

### Recommended

- **Encoder models** (BERT, DistilBERT, RoBERTa)
- **Long sequences** ($N \geq 256$)
- **Smooth attention patterns**

### Not Recommended

- **Decoder models** (no benefit, uses fallback)
- **Size reduction** (use compression methods)
- **Sharp attention patterns** (DCT truncation loses high-frequency info)

## 9.3 Decision Flowchart

```
Encoder or decoder? ─── Decoder ─── Use standard attention or other method
  │ Encoder
  ▼
Sequence > 256? ─── No ─── Galerkin benefit marginal
  │ Yes
  ▼
Use Galerkin
```

## 9.4 Cost-Benefit

For DistilBERT at $N = 256$:

| Metric | Baseline | Galerkin | Change |
|---|---|---|---|
| Attention FLOPs | 262K | 131K | **2× reduction** |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Params | 66.9M | {{D_PARAMS}}M | +{{D_PARAM_DELTA}}M |

## 9.5 Comparison with Published Methods

| Method | Encoder | Decoder | Exact |
|---|---|---|---|
| Linformer | $O(Nr)$ | $O(Nr)$ | Approx. |
| Performer | $O(Nm)$ | $O(Nm)$ | Approx. |
| FNet | $O(N\log N)$ | N/A | No |
| **Galerkin** | **$O(Nr)$** | **Fallback** | **Exact in basis** |

## 9.6 Deployment Considerations

### Encoder latency
2× faster at $N=128$, 8× at $N=512$.

### Decoder latency
No benefit.

### Memory
Encoder: $O(Nr)$ vs $O(N^2)$. Substantial savings for long sequences.

## 9.7 Open Questions

1. **Learned basis** — would a data-adaptive basis beat DCT?
2. **Causal Galerkin** — is there a causal-compatible basis?
3. **Higher-order DCT** — does adding derivatives help?
4. **Cross-attention** — does Galerkin work for encoder-decoder cross-attention?

## 9.8 Broader Impact

### Positive
- Encoder efficiency
- Mathematical elegance
- Composable with quantization

### Negative
- No benefit for decoders
- Parameter overhead
- Dual-use

### Environmental
Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.
