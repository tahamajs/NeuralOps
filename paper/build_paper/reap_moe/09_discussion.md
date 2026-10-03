# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: MoE Increases Capacity

REAP-MoE with $E = 8$ experts provides ~8× effective FFN capacity, at ~2× 
parameter cost due to SVD compression.

### 9.1.2 Takeaway 2: SVD Reduces Bloat

Standard MoE with $E = 8$ would 8× FFN params. SVD reduces this to ~1.5×.

### 9.1.3 Takeaway 3: Load-Balancing Works

The auxiliary loss keeps expert utilization balanced within 0.85×.

## 9.2 When to Use REAP-MoE

### Recommended
- **Capacity-scaling applications** where parameter count is not critical
- **Long-context training** where expert specialization helps
- **Multi-task learning** where experts specialize by task

### Not Recommended
- **Parameter-constrained deployment**
- **Latency-critical** applications (MoE adds routing overhead)
- **Small models** (<10M params; router overhead dominates)

## 9.3 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | REAP-MoE | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | 66.4M |
| Accuracy | 90.71% | 90.02% | -0.69% |
| VRAM | 183 MB | 554 MB MB | 380 MB |
| Time | 1.90 s | 1.42x s | 2.3 s× |

## 9.4 Comparison to Published MoE

| Method | Router | Experts | Compression |
|---|---|---|---|
| Shazeer (2017) | Noisy top-k | Dense | None |
| Switch | Top-1 | Dense | None |
| GShard | Top-2 | Dense | None |
| **REAP-MoE** | **Noisy top-k** | **SVD** | **~4×/expert** |

## 9.5 Deployment Considerations

### Storage
REAP-MoE increases storage by ~2× vs baseline. Not suitable for edge 
deployment without further compression.

### Latency
Router + expert selection adds ~10-20% overhead.

### Memory
All experts must be loaded, so VRAM scales with $E$.

## 9.6 Open Questions

1. **Adaptive $E$:** Can we learn optimal $E$ per layer?
2. **Expert pruning:** Can we prune underutilized experts?
3. **Cross-task transfer:** Do experts transfer between tasks?
4. **Interaction with quantization:** Can MoE be quantized?

## 9.7 Broader Impact

### Positive
- Increases model capacity without proportional compute
- Enables expert specialization
- Relevant for multi-task learning

### Negative
- Parameter growth may not be acceptable
- Complexity increases deployment burden

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---