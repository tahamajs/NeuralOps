<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

MinimalSVD reduces attention parameters by **{{D_RATIO}}×–{{G_RATIO}}×** with 
accuracy drops of **{{D_DELTA}}%–{{G_DELTA}}%**. Genuine reduction in model 
size.

### 9.1.2 Takeaway 2: No Calibration Needed

Unlike A3 and AASVD, MinimalSVD is **training-free**. Rank selection is based 
purely on the weight spectrum.

### 9.1.3 Takeaway 3: Minimal Baseline

MinimalSVD is the **shortest auditable implementation** of the energy-adaptive 
family. It is designed as a reference for researchers who want to extend or 
compare against the more complex variants.

## 9.2 When to Use MinimalSVD

### Recommended

- **Research baseline** for energy-adaptive SVD compression
- **Storage-constrained deployment**
- **Calibration data unavailable**
- **When code auditability is a priority**

### Not Recommended

- **Production deployment requiring pre-test safety** → use MinimalPRISM
- **Complex model families** requiring whole-model traversal → use NOVA/PRISM
- **Extremely small models** (<10M params)

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Continue
  │ No → other approaches

  │
  ▼
Production deployment? ─── Yes ─── Use MinimalPRISM (pre-test + safety)
  │ No
  ▼
Research baseline? ─── Yes ─── Use MinimalSVD (minimal code)
  │ No
  ▼
Use MinimalNOVA or MinimalPRISM
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | MinimalSVD | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | {{D_DISK}} MB | **{{D_DISK_RATIO}}×** |
| Params | 66.95M | {{D_PARAMS}}M | **{{D_RATIO}}×** |
| VRAM | 183 MB | {{D_VRAM}} MB | **{{D_VRAM_RATIO}}×** |
| Time | 1.90 s | {{D_TIME}} s | **{{D_TIME_RATIO}}×** |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |

## 9.5 Comparison with Published Methods

| Method | Calibration | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT (from scratch) | — | 3.5× | 89.4% |
| DistilBERT (distill) | — | 1.6× | 90.7% |
| A3 | Required | {{A3_RATIO}}× | {{A3_ACC}}% |
| AASVD | Required | {{AASVD_RATIO}}× | {{AASVD_ACC}}% |
| MinimalNOVA | None | {{NOVA_RATIO}}× | {{NOVA_ACC}}% |
| MinimalPRISM | None | {{PRISM_RATIO}}× | {{PRISM_ACC}}% |
| **MinimalSVD** | **None** | **{{D_RATIO}}×** | **{{D_ACC}}%** |

## 9.6 Deployment Considerations

### Storage
For 100M params:
- Baseline: 200 MB FP16
- MinimalSVD (r=128): ~{{D_DISK}} MB
- MinimalSVD + INT8: ~{{D_DISK_INT8}} MB

### Latency
MinimalSVD is faster than dense attention on GPU due to fewer FLOPs.

### Batching
Larger batches benefit more due to better cache utilization.

## 9.7 Open Questions

1. **Interaction with quantization**
2. **Rank schedule for generation tasks**
3. **Cross-lingual generalization**

## 9.8 Broader Impact

### Positive
- Democratizes AI
- Reduces carbon footprint
- Enables edge AI

### Negative
- Dual-use concerns
- Concentration risk

### Environmental
Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.

---