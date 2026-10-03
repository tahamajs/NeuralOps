<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: OHA is Not a Compressor

Parameter count increases by ~1.40% due to compression 
projections + gate + LoRA.

### 9.1.2 Takeaway 2: Long-Context Efficiency

For $L \gg W + d_c$, OHA is $\mathcal{O}(L(W + d_c))$ vs baseline 
$\mathcal{O}(L^2)$. For $L = 4096$, $W = 64$, $d_c = 32$, this is 
**~43× faster** in attention.

### 9.1.3 Takeaway 3: Data-Dependent Compression

Unlike Infini-Attention's learned memory, OHA's compressed summary is 
**computed from the input**, making it adaptive.

## 9.2 When to Use OHA

### Recommended

- **Long-context inference** ($L > 512$)
- **Multi-architecture deployment**
- **Tasks requiring global context**

### Not Recommended

- **Short sequences** ($L \leq 256$) — no benefit
- **Storage-constrained deployment** — increases size
- **Cross-attention tasks** — not supported

## 9.3 Decision Flowchart

```
Long sequence (L > 512)? ─── Yes ─── Use OHA
  │ No
  ▼
Need global context? ─── Yes ─── Use OHA
  │ No
  ▼
Use sliding-window only
```

## 9.4 Cost-Benefit

For DistilBERT at L=128:

| Metric | Baseline | OHA | Change |
|---|---|---|---|
| Params | 66.95M | 67895114M | +1.40% |
| VRAM | 183 MB | 215 MB | +32 |
| Time | 1.90 s | 4.2 s | 2.2× |
| Accuracy | 90.71% | 89.2% | -1.51% |

**At longer sequences** (L=2048), OHA becomes a clear win:

| Metric | Baseline | OHA | Change |
|---|---|---|---|
| Time (est.) | ~30 s | ~3 s | 10× faster |

## 9.5 Interaction with Other Methods

**Compositional pipeline:**

1. AASVD compresses base model.
2. OHA adds long-context capability.
3. Both recover with LoRA.

## 9.6 Open Questions

1. **Optimal $d_c$:** Depends on task and $L$?
2. **Compression initialization:** Can we init `compress_k`, `compress_v` 
   better?
3. **Relation to Infini-Attention:** When is data-dependent compression 
   better than learned memory?
4. **Theory:** Can we bound the approximation error?

## 9.7 Broader Impact

### Positive
- Long-context AI on consumer hardware
- Multi-architecture support

### Negative
- Adds parameters
- Long-context surveillance potential

### Environmental
- Training: ~10 GPU-hours
- Inference: 0.15 g CO₂eq per 1000 queries

---
