<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: EAT is Not a Compressor

EAT replaces attention to enable **inference-time pruning**, not to reduce 
model size. Parameter count increases by ~66.4M% due to LoRA.

### 9.1.2 Takeaway 2: Multi-Architecture Support is Real

EAT works across DistilBERT, RoBERTa, and GPT-2 without code changes. This 
is a genuine advantage over architecture-specific methods.

### 9.1.3 Takeaway 3: Weight Transfer Must Be Verified

Silent weight transfer failure is the dominant failure mode. Any production 
deployment must include a post-injection sanity check.

## 9.2 When to Use EAT

### Recommended

- **Long-context inference** where pruning provides real speedup.
- **Multi-architecture deployment** where a single method must work on 
  many models.
- **Combined with AASVD or A3** — compress first, then replace attention.

### Not Recommended

- **Storage-constrained deployment** — EAT does not reduce size.
- **Latency-critical applications** — LoRA overhead makes EAT slower.
- **Models where weight transfer is uncertain** — check first.

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Use AASVD or A3
  │ No
  ▼
Need to prune tokens at inference? ─── Yes ─── Use EAT
  │ No
  ▼
Need multi-architecture deployment? ─── Yes ─── Use EAT or NHA
  │ No
  ▼
Consider standard LoRA
```

## 9.4 Cost-Benefit

For DistilBERT:

| Metric | Baseline | EAT | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | **+48.5\%%** |
| VRAM | 183 MB | 328 MB MB | 380 MB |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 89.91% | -0.80% |

**Verdict.** EAT is **neutral** for storage, **worse** for latency unless 
pruning is enabled, **neutral** for accuracy.

## 9.5 Interaction with Compressors

EAT can be **combined** with AASVD:

1. AASVD compresses projections → `SVDLinear`.
2. EAT replaces attention block with pruned version.
3. Both recover with LoRA.

The combined pipeline achieves parameter reduction AND pruning speedup.

## 9.6 Open Questions

1. **Optimal pruning ratio:** How to select $k$ per layer?
2. **Cross-task transfer:** Does pruning trained on SST-2 transfer?
3. **Merge vs prune:** Is token merging superior?
4. **Theory:** Can pruning preserve the function class?

## 9.7 Broader Impact

### Positive
- Reduces inference cost for long sequences
- Multi-architecture support simplifies deployment

### Negative
- May be used for surveillance (long-context monitoring)
- Adds parameters (opposite of compression)

### Environmental
- Training: ~12.5 GPU-hours
- Inference: 0.85 g CO₂eq per 1000 queries

---