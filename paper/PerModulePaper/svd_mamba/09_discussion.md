<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Not a Compressor

SVD-Mamba increases parameter count by ~66.4M%.

### 9.1.2 Takeaway 2: New Capability

SVD-Mamba provides an SSM path for linear-time recurrence, which may 
benefit long-context tasks.

### 9.1.3 Takeaway 3: Gate Init is Critical

The $-7$ bias keeps the SSM silent. Changing this destabilizes training.

## 9.2 When to Use SVD-Mamba

### Recommended

- **Research exploration** of hybrid attention-SSM
- **Long-context tasks** (potential benefit)
- **Multi-architecture deployment**

### Not Recommended

- **Compression-constrained deployment**
- **Short-sequence tasks**
- **Latency-critical applications**

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Use AASVD or ASVD
  │ No
  ▼
Want to explore hybrid SSM? ─── Yes ─── Use SVD-Mamba
  │ No
  ▼
Need long-context? ─── Yes ─── Consider SVD-Mamba
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT at L=128:

| Metric | Baseline | SVD-Mamba | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | +48.5\%% |
| VRAM | 183 MB | 639 MB MB | 380 MB |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 50.92% | -39.79% |

**Verdict.** SVD-Mamba is not cost-effective at short sequences.

## 9.5 Open Questions

1. **Is the SSM path useful?** Does the model actually use it?
2. **Can the SSM be pre-trained?** Would pretraining the SSM branch 
   help?
3. **Does gating reach a meaningful value?** Or does it stay at 0.001?
4. **Interaction with AASVD:** Can we compress the SSM branch too?

## 9.6 Broader Impact

### Positive
- Advances hybrid attention-SSM research
- Multi-architecture support

### Negative
- Adds parameters
- Complex to deploy

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg

---