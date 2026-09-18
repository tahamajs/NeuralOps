# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Linear Encoders

For encoders (DistilBERT, RoBERTa), Linformer achieves $\mathcal{O}(L k)$ 
complexity with modest accuracy loss.

### 9.1.2 Takeaway 2: Decoder Fallback

For decoders (GPT-2), the causal fallback prevents correctness issues but 
sacrifices linearity.

### 9.1.3 Takeaway 3: Diagonal Init Matters

Our diagonal init converges dramatically faster than random init.

## 9.2 When to Use This Implementation

### Recommended
- **Encoder-only models** (BERT-family)
- **Long-sequence encoding** ($L \gg k$)
- **Parameter-tolerant deployments**

### Not Recommended
- **Short sequences** ($L \leq k$): no compression
- **Long-sequence decoders**: fallback negates benefit
- **Parameter-critical**: +7% overhead

## 9.3 Decision Flowchart

```
Model is encoder? ─── Yes ─── Use Linformer
  │ No
  ▼
L ≤ k? ─── Yes ─── No benefit
  │ No
  ▼
Accept quadratic fallback? ─── Yes ─── Use Linformer
  │ No
  ▼
Consider other methods (LEKAv3, etc.)
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 512$:

| Metric | Baseline | Linformer |
|---|---|---|
| Params | 66.95M | 66.4MM (+7%) |
| Attention FLOPs | 100% | 2.1\times% |
| Inference time | 1.9 s | 1.42x s |
| Accuracy | 90.71% | 50.92% |

## 9.5 Open Questions

1. **Learned max_seq_len:** Can $L_\text{max}$ adapt?
2. **Per-head k:** Can different heads use different $k$?
3. **Causal Linformer:** Is there a causal-safe projection?
4. **Interaction with quantization:** INT8 + Linformer?

## 9.6 Broader Impact

### Positive
- Reduces compute for long-context encoders
- Enables BERT-family models on small GPUs

### Negative
- Adds parameters
- No benefit for decoders

---