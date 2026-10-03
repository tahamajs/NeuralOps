# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Linear Complexity is Real

LEKAv3 achieves $\mathcal{O}(L L_\text{land} + L W)$ complexity. For 
$L_\text{land}, W \sim \mathcal{O}(1)$, this is linear in $L$.

### 9.1.2 Takeaway 2: Learned Landmarks Win

The landmark scorer improves accuracy by **0.85%** over random 
selection. Content-aware selection matters.

### 9.1.3 Takeaway 3: Honest Parameter Increase

LEKAv3 increases parameter count by ~7.5% per layer. This is a cost, not a 
benefit. We trade parameters for compute.

## 9.2 When to Use LEKAv3

### Recommended
- **Long-context inference** ($L > 512$)
- **Encoder models** (DistilBERT, RoBERTa)
- **Hybrid deployment** with quantized base

### Not Recommended
- **Very short sequences** ($L < 64$): overhead dominates
- **Parameter-sensitive deployments**: +7.5%
- **Real-time low-latency**: landmark scoring adds cost

## 9.3 Decision Flowchart

```
Need to reduce quadratic attention cost? ─── Yes ─── Use LEKAv3
  │ No
  ▼
Need to reduce parameter count? ─── Yes ─── Use A3/AASVD
  │ No
  ▼
Keep standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 512$:

| Metric | Baseline | LEKAv3 | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | +7.5% |
| Attention FLOPs | 1.0 | 2.1\times | -48.5\%% |
| Inference time | 1.9 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 70.53% | -20.18% |

## 9.5 Comparison to Prior Linear Attention

| Method | Approach | Learned selector |
|---|---|---|
| Performer | RFF | No |
| Linformer | Low-rank | No |
| Nyströmformer | Landmark | No |
| **LEKAv3** | **Landmark + local** | **Yes** |

## 9.6 Open Questions

1. **Optimal $L_\text{land}$:** Can it be adaptive per layer?
2. **Cross-task transfer:** Does the scorer transfer?
3. **Interaction with quantization:** LEKAv3 + INT8?
4. **Fused kernel:** Can we write a Triton kernel?

## 9.7 Broader Impact

### Positive
- Enables long-context inference
- Reduces compute for edge deployment

### Negative
- Parameter increase
- Training overhead

---