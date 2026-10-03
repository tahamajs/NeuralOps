# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: SSM-Inspired, Not True SSM

We provide an SSM-inspired gating mechanism with learned exponential decay. 
This is **not** true Mamba — we do not perform a selective scan, and the 
complexity remains $\mathcal{O}(L^2 d)$.

### 9.1.2 Takeaway 2: Weight Reuse Works

Despite the architectural difference, reusing pretrained Q/K/V/O weights is 
possible and effective. The two-branch formulation maps them to SSM-like 
inputs.

### 9.1.3 Takeaway 3: Preservation Gate is Critical

The preservation gate prevents the SSM output from initially dominating and 
destabilizing training.

### 9.1.4 Takeaway 4: Decay is Learnable

The per-dimension decay parameters learn meaningful time constants that 
correlate with layer depth.

## 9.2 When to Use This Module

### Recommended
- **As an SSM-inspired baseline** for comparison
- **When pretrained weight reuse is required**
- **When complexity is not critical**

### Not Recommended
- **Long-sequence inference**: quadratic memory
- **When true linear complexity is needed**: use Mamba
- **When the docstring's O(L) claim matters**: it's incorrect

## 9.3 Decision Flowchart

```
Want a pretrained-friendly SSM-inspired attention? ─── Yes ─── Use this
  │ No
  ▼
Need true linear complexity? ─── Yes ─── Use Mamba or other SSM
  │ No
  ▼
Keep standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at L = 128:

| Metric | Baseline | Mamba SSM |
|---|---|---|
| Params | 66.95M | 66.4MM |
| Accuracy | 90.71% | 73.62% |
| Inference time | 1.9 s | 1.42x s |
| Peak VRAM | 183 MB | 329 MB MB |

## 9.5 Comparison with Published SSMs

| Method | Complexity | Pretrained | Accuracy |
|---|---|---|---|
| S4 | $\mathcal{O}(L)$ | No | — |
| Mamba | $\mathcal{O}(L)$ | No | — |
| RWKV | $\mathcal{O}(L)$ | No | — |
| **Ours** | $\mathcal{O}(L^2 d)$ | **Yes** | **73.62%** |

## 9.6 Open Questions

1. **True O(L) implementation:** Can we avoid materializing the $[L, L, d]$ 
   weight tensor?
2. **Selective decay:** Can $\gamma$ depend on input?
3. **Cross-architecture transfer:** Does the learned decay transfer?
4. **Combination with Mamba scan:** Hybrid approach?

## 9.7 Broader Impact

### Positive
- Provides a pretrained-friendly SSM baseline
- Weight reuse enables fine-tuning from existing checkpoints

### Negative
- Quadratic complexity limits long-context use
- Name "Mamba" may be misleading

---