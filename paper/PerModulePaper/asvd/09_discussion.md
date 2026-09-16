<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

ASVD reduces parameter count by {{D_RATIO}}×–{{G_RATIO}}× with modest 
accuracy drops ({{D_DELTA}}%–{{G_DELTA}}%).

### 9.1.2 Takeaway 2: Explicit Σ is Valuable

Keeping $\Sigma$ as a separate parameter enables:
- Spectral inspection
- Rank truncation without re-factorization
- Direct manipulation

### 9.1.3 Takeaway 3: Energy-Based Rank Adapts

Per-layer rank selection via energy thresholding outperforms uniform rank.

## 9.2 When to Use ASVD

### Recommended

- **Storage-constrained deployment**
- **Multi-architecture deployment**
- **Base for further compression** (prune after ASVD)
- **Combined with quantization**

### Not Recommended

- **Extremely small models** (<10M params)
- **Extremely large models** (>10B params)
- **Latency-critical applications** (LoRA overhead)

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Use ASVD
  │ No
  ▼
Need per-layer rank control? ─── Yes ─── Use ASVD
  │ No
  ▼
Use AASVD or A3
```

## 9.4 Cost-Benefit

For DistilBERT:

| Metric | Baseline | ASVD | Change |
|---|---|---|---|
| Params | 66.95M | {{D_PARAMS}}M | **{{D_RATIO}}×** |
| Disk | 134 MB | {{D_DISK}} MB | **{{D_DISK_RATIO}}×** |
| Inference VRAM | 183 MB | {{D_VRAM}} MB | {{D_VRAM_RATIO}}× |
| Inference time | 1.90 s | {{D_TIME}} s | {{D_TIME_RATIO}}× |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |

## 9.5 Comparison with Published ASVD Paper

The published ASVD paper (arXiv:2307.07382) reports compression ratios of 
1.5-2× on LLaMA-7B. Our implementation achieves {{D_RATIO}}× on DistilBERT 
by using:

- Lower ranks ($r = 64$ vs $r \approx 128$)
- Dynamic rank selection
- Aggressive LoRA recovery

## 9.6 Interaction with Other Methods

**Compositional pipeline:**

1. **ASVD compresses** base weights → smaller model.
2. **Quantize** (INT8) → 2× further reduction.
3. **LoRA recovery** fine-tunes.

## 9.7 Open Questions

1. **Optimal ε:** Task-dependent?
2. **Full covariance:** Would AASVD's approach help ASVD?
3. **Cross-task transfer:** Do ranks transfer from SST-2 to MNLI?
4. **Quantization interaction:** INT8 + ASVD = ?

## 9.8 Broader Impact

### Positive
- Democratizes AI
- Reduces carbon footprint
- Enables edge AI

### Negative
- Dual-use
- Concentration risk

### Environmental
- Training: ~{{COMPUTE_HOURS}} GPU-hours
- CO₂: ~{{CO2}} kg

---