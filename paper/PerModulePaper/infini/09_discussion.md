<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Infini-Attention is Not a Compressor

Parameter count increases by ~48.5\%% due to memory + LoRA.

### 9.1.2 Takeaway 2: Long-Context Efficiency

For $L \gg W + d_c$, Infini-Attention is $\mathcal{O}(L(W + d_c))$ vs 
baseline $\mathcal{O}(L^2)$. For $L = 4096$, $W = d_c = 128$, this is 
**16× faster** in attention.

### 9.1.3 Takeaway 3: β Init is Critical

$\beta = -5$ preserves pretrained behavior at start. $\beta = 0$ gives 
50/50 fusion and destabilizes training.

## 9.2 When to Use Infini-Attention

### Recommended

- **Long-context inference** ($L > 512$)
- **Multi-architecture deployment**
- **Combined with AASVD** for both compression and long context

### Not Recommended

- **Short sequences** ($L \leq 256$) — no benefit
- **Storage-constrained deployment** — increases size
- **Cross-attention tasks** — not supported

## 9.3 Decision Flowchart

```
Long sequence (L > 512)? ─── Yes ─── Use Infini-Attention
  │ No
  ▼
Need to reduce model size? ─── Yes ─── Use AASVD or A3
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at L=128:

| Metric | Baseline | Infini | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | +48.5\%% |
| VRAM | 183 MB | 353 MB MB | 380 MB |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 90.25% | -0.46% |

**At longer sequences** (L=2048), Infini-Attention becomes a clear win:

| Metric | Baseline | Infini | Change |
|---|---|---|---|
| Time (est.) | ~30 s | ~3 s | 10× faster |

## 9.5 Interaction with Other Methods

**Compositional pipeline:**

1. AASVD compresses base model → smaller
2. Infini-Attention adds long-context → faster at long L
3. Both recover with LoRA

## 9.6 Open Questions

1. **Optimal $d_c$:** Depends on task and $L$?
2. **β schedule:** Should β be scheduled during training?
3. **Multiple memories:** Can we have $K$ memory matrices?
4. **Interpretability:** What does the memory learn?

## 9.7 Broader Impact

### Positive
- Enables long-context AI on consumer hardware
- Multi-architecture support simplifies deployment

### Negative
- Adds parameters (opposite of compression)
- Long-context could enable mass surveillance

### Environmental
- Training: ~12.5 GPU-hours
- Inference: 0.85 g CO₂eq per 1000 queries

---