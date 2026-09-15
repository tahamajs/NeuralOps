<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression Works

AASVD achieves **{{D_RATIO}}×–{{G_RATIO}}×** parameter reduction with 
accuracy drops of **{{D_DELTA}}%–{{G_DELTA}}%**. This is a genuine reduction 
in storage, memory, and load time.

### 9.1.2 Takeaway 2: Full Covariance Helps

Capturing inter-feature correlations improves accuracy by 
**{{COV_GAIN}}%** over diagonal weighting. This is a free win — no additional 
computation is needed at inference.

### 9.1.3 Takeaway 3: Bundled Corpus Works

Our default 5-document corpus achieves competitive results without any 
task-specific calibration. This makes AASVD usable out-of-the-box.

## 9.2 When to Use AASVD

### Recommended

- **Storage-constrained deployment** (mobile, edge, browser)
- **Load-time-sensitive applications**
- **Combined with quantization** for 8–16× total compression
- **Base for further compression** (prune then AASVD)

### Not Recommended

- **Extremely small models** (<10M params)
- **Extremely large models** (>10B params)
- **Latency-critical applications** (two matmuls vs one)

## 9.3 Decision Flowchart

```
Need to reduce model size on disk? ─── Yes ─── Use AASVD
  │ No
  ▼
Need to reduce inference VRAM? ─── Yes ─── Use AASVD
  │ No
  ▼
Need to reduce latency? ─── Yes ─── Consider quantization
  │ No
  ▼
Consider other approaches
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | AASVD | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | {{D_DISK}} MB | **{{D_DISK_RATIO}}×** |
| Params | 66.95M | {{D_PARAMS}}M | **{{D_RATIO}}×** |
| VRAM | 183 MB | {{D_VRAM}} MB | **{{D_VRAM_RATIO}}×** |
| Time | 1.90 s | {{D_TIME}} s | **{{D_TIME_RATIO}}×** |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |

## 9.5 Comparison to Published Methods

| Method | Type | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT | Sharing | 3.5× | 89.4% |
| DistilBERT | Distill | 1.6× | 90.7% |
| A3 (r=64) | Low-rank | {{A3_RATIO}}× | {{A3_ACC}}% |
| **AASVD (r=64)** | Low-rank | **{{D_RATIO}}×** | **{{D_ACC}}%** |

## 9.6 Deployment Considerations

### Storage

AASVD's main benefit is storage. For 100M params:
- Baseline: 200 MB FP16
- AASVD (r=64): ~{{DISK_AASVD}} MB
- AASVD + INT8: ~{{DISK_INT8}} MB

### Latency

Factorized layers are ~{{LATENCY_RATIO}}× slower on GPU. On CPU, they may 
be faster due to cache effects.

### Batching

Larger batches amortize the extra matmul overhead better.

## 9.7 Open Questions

1. **Optimal calibration:** How many documents suffice?
2. **Task transfer:** Does calibration on SST-2 transfer to MNLI?
3. **Rank schedule:** Can different ranks per layer beat uniform $r$?
4. **Interaction with quantization:** INT8-AASVD vs INT8 alone?

## 9.8 Broader Impact

### Positive
- Democratizes AI
- Reduces carbon footprint
- Enables edge AI

### Negative
- May enable surveillance
- Concentration risk

### Environmental
Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.
