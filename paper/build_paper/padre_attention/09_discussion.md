# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Polynomial Substitution Works

The Taylor approximation of softmax is accurate enough for pretrained 
transformers with orders ≥ 2.

### 9.1.2 Takeaway 2: Centering is Critical

Without max-centering, the polynomial kernel overflows in FP16. This is 
the most important engineering detail.

### 9.1.3 Takeaway 3: Kernelization is Left to Future Work

The module is **kernelizable in theory**, but not in this implementation. 
Runtime is still $\mathcal{O}(N^2 d_h)$.

### 9.1.4 Takeaway 4: Order 2 is a Good Default

Higher orders give diminishing returns.

## 9.2 When to Use PADRe

### Recommended
- As a baseline for polynomial attention research
- When testing kernelization strategies
- When dropout regularization on attention is desired

### Not Recommended
- When runtime speedup is the goal (no kernelization)
- When exact softmax is required
- When $|s| > 5$ (polynomial breaks down)

## 9.3 Decision Flowchart

```
Want polynomial attention? ─── Yes ─── Use PADRe
  │ No
  ▼
Want softmax alternative? ─── Yes ─── Consider Softpick
  │ No
  ▼
Keep standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 128$:

| Metric | Baseline | PADRe |
|---|---|---|
| Params | 66.95M | 66.4MM |
| Attention FLOPs | 100% | 100% |
| Inference time | 1.9 s | 1.42x s |
| Accuracy | 90.71% | 90.37% |

## 9.5 Comparison with Published Methods

| Method | Kernel | Kernelized | Pretrained |
|---|---|---|---|
| Softmax | exp | ✗ | — |
| Performer | RFF | ✓ | ✗ |
| Linear Transformer | elu+1 | ✓ | ✗ |
| **PADRe** | **poly** | **Partial** | **✓** |

## 9.6 Open Questions

1. **Kernelization:** What's the practical feature dimension?
2. **Order selection:** Learned or fixed?
3. **Interaction with GQA:** Does kernelization preserve GQA?
4. **Cross-architecture transfer:** Does order 2 work everywhere?

## 9.7 Broader Impact

### Positive
- Provides polynomial attention baseline
- Multi-architecture support
- Dropout regularization on attention

### Negative
- No actual speedup
- Kernelization deferred

---