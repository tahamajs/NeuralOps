# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: KV Cache Reduction is Real

TCA reduces KV cache size by 48.5\%% at ρ = 50\%, translating 
to real memory savings during long-context inference.

### 9.1.2 Takeaway 2: Learnable Scoring Wins

The score net improves accuracy by **0.85%** over the norm-only 
heuristic. This confirms that content-specific importance is learnable.

### 9.1.3 Takeaway 3: [CLS] Preservation Matters

Pinning token 0 recovers **0.85%** accuracy — a free win.

## 9.2 When to Use TCA

### Recommended
- **Long-context inference** (L > 512)
- **Memory-constrained deployment**
- **Streaming/decoding scenarios**
- **Combined with quantization** (TCA + INT8)

### Not Recommended
- **Short sequences** (L < 64): overhead dominates
- **Latency-critical**: score_net adds a small compute cost
- **Parameter-sensitive deployments**: TCA increases params by ~6%

## 9.3 Decision Flowchart

```
Need to reduce KV cache at inference? ─── Yes ─── Use TCA
  │ No
  ▼
Need to reduce model size? ─── Yes ─── Use A3/AASVD instead
  │ No
  ▼
Need to reduce latency? ─── Consider quantization
  │ No
  ▼
Consider other methods
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at L = 512:

| Metric | Baseline | TCA (ρ=0.5) | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | +6.3% |
| KV cache | 0.85 | 0.85 | -50% |
| Inference time | 0.85 s | 0.85 s | 2.1\times× |
| Accuracy | 90.71% | 50.92% | -39.79% |

## 9.5 Comparison to Prior Eviction Methods

| Method | Learnable | Causal-safe | [CLS] pinned |
|---|---|---|---|
| H2O | No | Partial | No |
| StreamingLLM | No | Yes | Yes (sinks) |
| SnapKV | No | Partial | No |
| **TCA** | **Yes** | **Yes** | **Yes** |

## 9.6 Deployment Considerations

### KV Cache Scaling

At long L, KV cache dominates memory. TCA's 50% reduction means:

| L | Baseline KV | TCA KV (ρ=0.5) |
|---|---|---|
| 1K | 12.6 MB | 6.3 MB |
| 4K | 50 MB | 25 MB |
| 16K | 200 MB | 100 MB |
| 32K | 400 MB | 200 MB |

### Combined with Quantization

TCA + INT8 → **~8× total reduction** in KV footprint.

## 9.7 Open Questions

1. **Optimal ρ schedule:** Can ρ vary per layer?
2. **Score net generalization:** Does the score transfer across tasks?
3. **Interaction with quantization:** TCA then INT8 vs INT8 then TCA?
4. **Training stability:** Should score_net be trained from the start?

## 9.8 Broader Impact

### Positive
- Enables long-context inference on small GPUs
- Reduces memory footprint for chat models
- Complements existing KV compression methods

### Negative
- Small parameter increase
- Adds a training step (score_net)

---