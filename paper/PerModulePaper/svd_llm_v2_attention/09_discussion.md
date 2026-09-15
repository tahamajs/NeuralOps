<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

SVD-LLM V2 reduces attention parameters by **{{D_RATIO}}×–{{G_RATIO}}×** with 
accuracy drops of **{{D_DELTA}}%–{{G_DELTA}}%**.

### 9.1.2 Takeaway 2: Interpretable Compression

Unlike symmetric-factorization methods, SVD-LLM V2 preserves the singular 
spectrum for inspection and downstream applications.

### 9.1.3 Takeaway 3: Multi-Architecture Hardening

Pre-test, injection-time nested detection, and GQA pooling make SVD-LLM V2 
robust across four families.

## 9.2 When to Use SVD-LLM V2

### Recommended

- **Downstream flexibility needed** (quantization, spectral analysis)
- **Production deployment** (pre-test and safety)
- **Multi-architecture models** (GQA + nested + causal)
- **Combined with quantization**

### Not Recommended

- **Extremely small models** (<10M params)
- **Strict FP16-only deployment** (consider symmetric)

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Continue
  │ No → other approaches

  │
  ▼
Need interpretable SVD factors? ─── Yes ─── SVD-LLM V2
  │ No
  ▼
Deployment in production? ─── Yes ─── SVD-LLM V2 or MinimalPRISM
  │ No
  ▼
Research baseline? ─── MinimalSVD
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | SVD-LLM V2 | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | {{D_DISK}} MB | **{{D_DISK_RATIO}}×** |
| Params | 66.95M | {{D_PARAMS}}M | **{{D_RATIO}}×** |
| VRAM | 183 MB | {{D_VRAM}} MB | **{{D_VRAM_RATIO}}×** |
| Time | 1.90 s | {{D_TIME}} s | **{{D_TIME_RATIO}}×** |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |

## 9.5 Comparison with Published Methods

| Method | Calibration | Ratio | SST-2 acc | Interpretable |
|---|---|---|---|---|
| ALBERT | — | 3.5× | 89.4% | No |
| DistilBERT | — | 1.6× | 90.7% | No |
| AWQ | Required | 4× | 90.5% | No |
| A3 | Required | 5× | {{A3_ACC}}% | No |
| MinimalPRISM | None | {{PRISM_RATIO}}× | {{PRISM_ACC}}% | No |
| **SVD-LLM V2** | **None** | **{{D_RATIO}}×** | **{{D_ACC}}%** | **Yes** |

## 9.6 Deployment Considerations

### Storage
For 100M params:
- Baseline: 200 MB FP16
- SVD-LLM V2 (r=128): ~{{D_DISK}} MB
- SVD-LLM V2 + INT8: ~{{D_DISK_INT8}} MB

### Latency
SVD-LLM V2 is faster than dense attention on GPU due to fewer FLOPs.

### Batching
Larger batches benefit more due to better cache utilization.

## 9.7 Open Questions

1. **Interaction with quantization:** can we quantize $\Sigma$ separately?
2. **Cross-task rank:** does the same rank work for generation tasks?
3. **Spectral transfer:** do spectra transfer across models?

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
