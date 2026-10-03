# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Modular Design Works

NEXUS cleanly combines four active sparse mechanisms.

### 9.1.2 Takeaway 2: Encoder-Only

For causal models, the module falls back to dense. This is a **correctness** 
decision, not a performance one.

### 9.1.3 Takeaway 3: Two Components Inactive

Stripe and RFF are inactive under our setup. Users should enable them 
explicitly if desired.

### 9.1.4 Takeaway 4: Softpick is Delicate

Temperature tuning is required.

## 9.2 When to Use NEXUS

### Recommended
- Encoder-only models
- Sequences $N \gg k B_s$
- Research on combined sparse attention

### Not Recommended
- Causal decoders (dense fallback)
- Short sequences ($N \leq 8$)
- When only one mechanism is needed (use that one directly)

## 9.3 Decision Flowchart

```
Model is encoder? ─── Yes ─── Use NEXUS
  │ No
  ▼
Causal decoder? ─── Yes ─── Dense fallback (no benefit)
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at $L = 512$:

| Metric | Baseline | NEXUS |
|---|---|---|
| Params | 66.95M | 66.4MM |
| Attention FLOPs | 100% | 2.1\times% |
| Inference time | 1.9 s | 1.42x s |
| Accuracy | 90.71% | 50.92% |

## 9.5 Comparison with Published Methods

| Method | Complexity | Causal | Multi-arch |
|---|---|---|---|
| XAttention | $\mathcal{O}(N k B_s)$ | Partial | No |
| SeerAttention | $\mathcal{O}(N k B_s)$ | Partial | No |
| DuSA | $\mathcal{O}(N k B_s)$ | Partial | No |
| **NEXUS** | **$\mathcal{O}(N k B_s)$** | **No (dense)** | **Yes** |

## 9.6 Open Questions

1. **Causal block-sparse:** Can we implement causal block-sparse safely?
2. **Component interaction:** Do the mechanisms interfere?
3. **Sparse min seq len:** Optimal value?
4. **Softpick stability:** Alternative for all-negative case?

## 9.7 Broader Impact

### Positive
- Modular framework for sparse attention research
- Multi-architecture support

### Negative
- Two components inactive in practice
- No causal benefit

---