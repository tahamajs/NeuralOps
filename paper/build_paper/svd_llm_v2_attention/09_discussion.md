<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

SVD-LLM V2 reduces attention parameters by **50\%×–2.1\times×** with 
accuracy drops of **-8.83%–-28.44%**.

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
| Disk (FP16) | 134 MB | 0.85 MB | **2.1\times×** |
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 289 MB MB | **2.1\times×** |
| Time | 1.90 s | 1.42x s | **2.1\times×** |
| Accuracy | 90.71% | 81.88% | -8.83% |

## 9.5 Comparison with Published Methods

| Method | Calibration | Ratio | SST-2 acc | Interpretable |
|---|---|---|---|---|
| ALBERT | — | 3.5× | 89.4% | No |
| DistilBERT | — | 1.6× | 90.7% | No |
| AWQ | Required | 4× | 90.5% | No |
| A3 | Required | 5× | 86.4\%% | No |
| MinimalPRISM | None | 2.1\times× | 86.4\%% | No |
| **SVD-LLM V2** | **None** | **50\%×** | **81.88%** | **Yes** |

## 9.6 Deployment Considerations

### Storage
For 100M params:
- Baseline: 200 MB FP16
- SVD-LLM V2 (r=128): ~0.85 MB
- SVD-LLM V2 + INT8: ~0.85 MB

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
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---