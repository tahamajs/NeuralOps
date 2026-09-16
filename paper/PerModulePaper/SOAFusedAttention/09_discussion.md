# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression, Zero-Shot

SOA Fused Attention **reduces** attention parameters by **{{D_RATIO}}×** 
without requiring any calibration data. This is a significant practical 
advantage over A3 and AASVD.

### 9.1.2 Takeaway 2: $W_O$ Should Stay

Factorizing $W_O$ would give marginal additional savings but compound error. 
We keep it intact.

### 9.1.3 Takeaway 3: Works Across Architectures

The same `copy_weights` handles DistilBERT, RoBERTa, GPT-2, and Qwen3.5.

## 9.2 When to Use SOA Fused Attention

### Recommended
- **Zero-shot deployment** where calibration data is unavailable.
- **Parameter-sensitive deployments** where every million counts.
- **Attention-heavy models** (DistilBERT, BERT-family).
- **Composing with GQA** for additional KV reduction.

### Not Recommended
- **Long-context inference** — attention is still $\mathcal{O}(L^2)$.
- **Models with small attention share** (e.g., large FFN expansion).
- **Very low rank** ($k < 64$) — reconstruction error too high.

## 9.3 Decision Flowchart

```
Have calibration data? ─── Yes ─── Consider A3 or AASVD
  │ No
  ▼
Want to reduce parameters? ─── Yes ─── Use SOA Fused
  │ No
  ▼
Consider attention-replacement methods
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at $k = 128$:

| Metric | Baseline | Ours | Change |
|---|---|---|---|
| Attention params | 2.36M | {{D_ATTN_PARAMS}} | **{{D_RATIO}}×** |
| Total params | 66.95M | {{D_TOTAL_PARAMS}} | {{D_TOTAL_DELTA}}% |
| Inference time | 1.90 s | {{D_TIME}} s | {{D_TIME_RATIO}}× |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |

## 9.5 Composition with Other Methods

SOA Fused Attention composes with:

- **GQA**: further reduces KV projection size.
- **KV quantization**: INT8 on the factorized projections.
- **Token compression** (TCA): reduces KV length after projection.
- **FFN compression** (A3/AASVD): compresses the other half of the block.

A combined pipeline could achieve **4-5× total compression** with the same 
accuracy recovery strategy.

## 9.6 Broader Impact

### Positive
- Enables deployment on smaller hardware.
- Zero-shot — no privacy concerns from calibration data.
- Simple to apply.

### Negative
- Attention remains quadratic — not a long-context solution.
- $W_O$ preserved — modest total reduction.

## 9.7 Open Questions

1. **Optimal $k$ scheduling:** Should different layers use different ranks?
2. **Interaction with LoRA placement:** Should LoRA be on down or up?
3. **Rank selection by singular value:** Can we auto-select $k$ per layer?
4. **Composition with KV compression:** How do they interact?

---