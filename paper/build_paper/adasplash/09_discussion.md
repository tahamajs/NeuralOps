<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Sparsity ≠ Compression

AdaSplash produces sparse attention but **increases** parameter count. 
It is not a compression method. Its value is in:
- **Interpretability** (sparse attention maps)
- **Potential compute savings** on sparse hardware
- **Attention pattern analysis**

### 9.1.2 Takeaway 2: α = 1.5 is a Good Default

Empirically, α = 1.5 balances sparsity and accuracy. α = 2.0 (sparsemax) 
is too sparse for stable training.

### 9.1.3 Takeaway 3: Attention Sinks Matter

Preserving the first 4 tokens is critical. Without them, accuracy drops by 
4.5%.

## 9.2 When to Use AdaSplash

### Recommended

- **Interpretability research** — sparse attention is easier to analyze
- **Attention pattern analysis** — sparsity reveals structure
- **Sparse hardware deployment** — 2:4 sparsity can be leveraged
- **Long-context inference** — splash masking reduces KV cache

### Not Recommended

- **Model size reduction** — AdaSplash increases size
- **Latency-critical inference** — bisection is slow
- **Extreme compression** — use A3/AASVD instead
- **Simple classification** — softmax already works

## 9.3 Decision Flowchart

```
Need size reduction? ─── Yes ─── Use A3/AASVD/Adaptive SVD
  │ No
  ▼
Need interpretable attention? ─── Yes ─── AdaSplash
  │ No
  ▼
Need latency reduction? ─── Yes ─── Use quantized attention
  │ No
  ▼
Consider AdaSplash for research
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on SST-2:

| Metric | Baseline | AdaSplash | Change |
|---|---|---|---|
| Accuracy | 90.71% | 77.18% | -13.53% |
| Params | 66.95M | 67549448M | +594438 |
| VRAM | 183 MB | 2526 MB | +12 |
| Inference time | 1.90 s | 3.5 s | 1.84× |
| Attention sparsity | 0% | 42% | +40% |

**Verdict.** AdaSplash trades size and latency for sparsity and interpretability.

## 9.5 Comparison with Published Methods

| Method | Type | Params | Sparse | Latency |
|---|---|---|---|---|
| Softmax | Baseline | 0% | No | 1× |
| Longformer | Local | 0% | No | ~1× |
| Performer | Kernel | +linear | No | ~1× |
| Sparsemax | Activation | 0% | Yes | ~5× |
| **AdaSplash** | **Activation + Local** | **+1%** | **Yes** | **~1.9×** |

## 9.6 Deployment Considerations

### Storage
AdaSplash increases storage slightly due to LoRA.

### Latency
~1.9× slower. Acceptable for research; not for production.

### Interpretability
Sparse attention maps can be visualized directly.

### Sparse Hardware
If the deployment target has 2:4 sparsity support (e.g., Ampere GPUs), 
AdaSplash's sparse outputs can be exploited for actual speedup.

## 9.7 Open Questions

1. **Fused kernels:** Would a Triton kernel for bisection speed it up?
2. **Learned α:** Would per-layer learned α help?
3. **Sparse hardware:** Actual speedup on Ampere 2:4 sparsity?
4. **Long context:** How does AdaSplash scale to L = 4096?

## 9.8 Broader Impact

### Positive
- **Interpretability:** Sparse attention for research
- **Attention analysis:** Easier to study patterns
- **Sparse hardware:** Potential for real speedup

### Negative
- **Not a compression method** — may mislead
- **Slower inference**
- **Dual-use**

### Environmental
Total compute: ~12 GPU-hours, ~12.4 kg CO₂eq.
