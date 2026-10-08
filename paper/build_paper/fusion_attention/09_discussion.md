# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Short-Path Exactness

For $L \leq 128$, FUSIONATTN produces **exactly** the pretrained model's 
output (with base weights copied and LoRA initially zero). This is a 
desirable property but means no sparsity is active.

### 9.1.2 Takeaway 2: Sparsity is Length-Dependent

FUSIONATTN's benefits only manifest at $L \geq 256$. Below that, the 
dispatch logic disables sparsity.

### 9.1.3 Takeaway 3: Not a Compression Method

Parameter count increases by LoRA + gate MLP. Not for size reduction.

## 9.2 When to Use FUSIONATTN

### Recommended

- **Long-context inference** ($L \geq 512$)
- **Content-dependent attention patterns**
- **Multi-paradigm research**

### Not Recommended

- **Short sequences** ($L < 256$) — no benefit, more parameters
- **Model size reduction** — use compression methods
- **Strict latency requirements** — gate adds overhead

## 9.3 Decision Flowchart

```
Need size reduction? ─── Yes ─── Use A3/AASVD
  │ No
  ▼
Sequence length > 256? ─── No ─── Use baseline (or LoRA-only)
  │ Yes
  ▼
Content-dependent sparsity needed? ─── No ─── Use Longformer
  │ Yes
  ▼
Use FUSIONATTN
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 512$:

| Metric | Baseline | FUSIONATTN | Change |
|---|---|---|---|
| Attention FLOPs | 262K | 65K | **4× reduction** |
| Accuracy | 90.71% | 86.4% | 0.85% |
| Params | 66.9M | 66.4MM | +-32MM |
| Gate overhead | 0 | ~32K params | +0.05% |

## 9.5 Comparison with Published Methods

| Method | Type | Complexity | Length-aware |
|---|---|---|---|
| Longformer | Fixed sparse | $O(LW)$ | No |
| Routing | Content sparse | $O(L^2/K)$ | No |
| MoBA | Block sparse | $O(KL)$ | No |
| **FUSIONATTN** | **Hybrid** | **$O(L^2)$ or $O(KL)$** | **Yes** |

## 9.6 Deployment Considerations

### Latency
- Short path: same as baseline.
- Long path: $K/B \times$ faster.

### Memory
- Short path: same.
- Long path: $O(K \cdot L \cdot B)$ instead of $O(L^2)$.

### Parameter Overhead
- +66.4M% via LoRA + gate.

## 9.7 Open Questions

1. **Optimal threshold** for various tasks?
2. **Gate training** — does it learn meaningful patterns?
3. **Very long context** ($L > 8192$) behavior?
4. **Combination with quantization**?

## 9.8 Broader Impact

### Positive
- Long-context efficiency
- Content-aware sparsity
- Composable with other efficiency methods

### Negative
- Parameter overhead
- Zero benefit at L ≤ 128
- Dual-use

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---