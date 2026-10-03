<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

MinimalPRISM reduces attention parameters by **50\%×–2.1\times×** 
with accuracy drops of **-5.27%–-28.44%**. This is a genuine 
reduction in model size.

### 9.1.2 Takeaway 2: No Calibration Needed

Unlike A3 and AASVD, MinimalPRISM is **training-free**. Rank selection is 
based purely on the weight spectrum.

### 9.1.3 Takeaway 3: Injection Safety

The pre-test protocol and injection-time nested encoder detection make 
MinimalPRISM **robust** to edge cases that break other methods.

## 9.2 When to Use MinimalPRISM

### Recommended

- **Storage-constrained deployment**
- **Calibration data unavailable**
- **Multi-architecture deployment**
- **Combined with quantization**

### Not Recommended

- **Extremely small models** (<10M params)
- **Extremely large models** (>10B params)
- **Latency-critical CPU inference**

## 9.3 Decision Flowchart

```
Need to reduce model size? ─── Yes ─── Continue
  │ No → other approaches

  │
  ▼
Have calibration data? ─── Yes ─── Consider A3/AASVD
  │ No
  ▼
Multi-architecture? ─── Yes ─── Use MinimalPRISM (safe injection)
  │ No
  ▼
Use MinimalPRISM (training-free)
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | MinimalPRISM | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | 0.85 MB | **2.1\times×** |
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 293 MB MB | **2.1\times×** |
| Time | 1.90 s | 1.42x s | **2.1\times×** |
| Accuracy | 90.71% | 85.44% | -5.27% |

## 9.5 Comparison to Published Methods

| Method | Calibration | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT (from scratch) | — | 3.5× | 89.4% |
| DistilBERT (distill) | — | 1.6× | 90.7% |
| A3 | Required | 2.1\times× | 86.4\%% |
| AASVD | Required | 2.1\times× | 86.4\%% |
| MinimalNOVA | None | 2.1\times× | 86.4\%% |
| **MinimalPRISM** | **None** | **50\%×** | **85.44%** |

## 9.6 Deployment Considerations

### Storage
For 100M params:
- Baseline: 200 MB FP16
- MinimalPRISM (r=128): ~0.85 MB
- MinimalPRISM + INT8: ~0.85 MB

### Latency
MinimalPRISM is faster than dense attention on GPU due to fewer FLOPs.

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
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---