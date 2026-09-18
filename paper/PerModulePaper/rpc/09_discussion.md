<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: FLOPs, Not Memory

RPC reduces attention FLOPs but not memory. The $L \times L$ score matrix 
is preserved.

### 9.1.2 Takeaway 2: Long Sequences Only

RPC's efficiency benefit appears at $L > 512$. Below that, overhead 
dominates.

### 9.1.3 Takeaway 3: Not a Compressor

RPC does not reduce parameters. LoRA adds ~1%.

## 9.2 When to Use RPC

### Recommended

- **Long-context inference** ($L > 512$)
- **Tasks where exact attention is not critical**
- **Combined with AASVD** for parameter + FLOP reduction

### Not Recommended

- **Short sequences** ($L \leq 256$)
- **Memory-constrained deployment**
- **Storage-constrained deployment**

## 9.3 Decision Flowchart

```
Long sequence (L > 512)? ─── Yes ─── Use RPC
  │ No
  ▼
Memory-constrained? ─── Yes ─── Use AASVD or EAT
  │ No
  ▼
Need parameter reduction? ─── Yes ─── Use AASVD or ASVD
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at L=128:

| Metric | Baseline | RPC | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | +48.5\%% |
| VRAM | 183 MB | 334 MB MB | 0% |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 87.84% | -2.87% |

**At L=1024**, RPC becomes beneficial:

| Metric | Baseline | RPC | Change |
|---|---|---|---|
| Time | ~15 s | ~8 s | 1.9× faster |

## 9.5 Interaction with Other Methods

**Compositional pipeline:**

1. **AASVD** compresses base weights → smaller model.
2. **RPC** sketches Q/K → fewer score FLOPs.
3. **LoRA recovery** fine-tunes both.

## 9.6 Open Questions

1. **Optimal $r$:** Task-dependent?
2. **Sparse projections:** Would Achlioptas-style sparse projections help?
3. **Learned projections:** Would they break JL?
4. **Long-context:** Actual speedup at L=4096?

## 9.7 Broader Impact

### Positive
- Reduces compute for long sequences
- Multi-architecture support

### Negative
- Does not reduce memory
- Adds parameters

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg

---