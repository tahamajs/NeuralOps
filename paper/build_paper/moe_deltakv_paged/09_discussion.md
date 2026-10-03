# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Combining Techniques Works

The three components (MoBA, Delta-KV, 4-bit quantization) compose cleanly 
with modest accuracy loss.

### 9.1.2 Takeaway 2: Simulation is Not Compression

The Delta-KV and 4-bit steps are **simulations**. They do not reduce 
inference memory. Users should be aware of this distinction.

### 9.1.3 Takeaway 3: Router Needs Careful Initialization

The identity initialization of $W_\text{rg}$ + load balancing loss prevents 
router collapse.

## 9.2 When to Use This Module

### Recommended
- **As a training-time regularization** for future quantization
- **When block-sparse attention is desired**
- **As a research baseline** for MoBA + Delta-KV combinations

### Not Recommended
- **When memory reduction is the goal**: no actual savings
- **When SVD-in-forward is expensive**: consider pre-computed deltas
- **When 6D gather OOM is a risk**: reduce page count

## 9.3 Decision Flowchart

```
Want block-sparse attention? ─── Yes ─── Use MoBA (simpler)
  │ No
  ▼
Want Delta-KV training-time reg? ─── Yes ─── Use this
  │ No
  ▼
Want actual compression? ─── Yes ─── Use A3/AASVD instead
  │ No
  ▼
Keep standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at L = 512:

| Metric | Baseline | MoE-DeltaKV |
|---|---|---|
| Params | 66.95M | 66.4MM |
| Attention FLOPs | 100% | 2.1\times% |
| Inference time | 1.9 s | 1.42x s |
| Accuracy | 90.71% | 86.47% |
| KV memory | unchanged | **unchanged** |

## 9.5 Comparison with Published Methods

| Method | Block-sparse | Delta | 4-bit | Real compression |
|---|---|---|---|---|
| MoBA | ✓ | ✗ | ✗ | ✗ |
| Delta-KV | ✗ | ✓ | ✗ | Partial |
| KVQuant | ✗ | ✗ | ✓ | ✓ |
| **Ours** | ✓ | ✓ | ✓ | ✗ (simulation) |

## 9.6 Open Questions

1. **Real compression:** Can the delta SVD be materialized?
2. **Adaptive page size:** Can pages vary in size?
3. **Router refinement:** Can the router attend to partial pages?
4. **Combined with quantization:** INT8 + Delta-KV?

## 9.7 Broader Impact

### Positive
- Provides a MoBA + Delta-KV baseline
- Demonstrates STE for temporal deltas

### Negative
- Memory savings are simulated
- 6D gather is memory-heavy

---