<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression is Possible

Unlike attention-replacement methods that preserve parameter count, A3 
achieves **{{D_RATIO}}×–{{G_RATIO}}×** parameter reduction with **<5%** 
accuracy drop. This is a genuine reduction in storage, memory, and load time.

### 9.1.2 Takeaway 2: Weighting Matters

Fisher weighting reduces reconstruction error by **{{FISHER_REDUCTION}}%** at 
the same rank. This is a free win — no additional computation beyond the 
calibration pass.

### 9.1.3 Takeaway 3: LayerNorm Recalibration is Cheap and Effective

A single recalibration pass, costing **{{LN_TIME}}s**, recovers 
**{{LN_RECOVERY}}%** of the accuracy drop on average.

## 9.2 When to Use A3

### 9.2.1 Recommended

- **Storage-constrained deployment:** Mobile, edge, browser.
- **Load-time-sensitive applications:** Cold-start latency matters.
- **Combined with quantization:** A3 + INT8 → 8× compression.
- **Base for further compression:** A3 then prune then quantize.

### 9.2.2 Not Recommended

- **Extremely small models:** <10M params; overhead dominates.
- **Extremely large models:** >10B; SVD time dominates.
- **Latency-critical applications:** Factorized layers are slower than dense 
  layers on GPUs (two matmuls instead of one).
- **Models with non-linear attention:** A3 only compresses linear layers.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Need to reduce model size on disk? ─── Yes ─── Use A3
  │
  No
  ▼
Need to reduce inference VRAM? ─── Yes ─── Use A3
  │
  No
  ▼
Need to reduce inference latency? ─── Yes ─── DON'T use A3
  │
  No
  ▼
Consider quantization instead
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | A3 | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | {{D_DISK}} MB | **{{D_DISK_RATIO}}×** |
| Params | 66.95M | {{D_PARAMS}}M | **{{D_RATIO}}×** |
| Inference VRAM | 183 MB | {{D_VRAM}} MB | **{{D_VRAM_RATIO}}×** |
| Inference time | 1.9 s | {{D_TIME}} s | **{{D_TIME_RATIO}}×** |
| Accuracy | 90.71% | {{D_ACC}}% | **{{D_DELTA}}%** |

**Verdict:** A3 is a **strict win** for storage and VRAM, **neutral** for 
latency, **small loss** for accuracy.

## 9.5 Comparison to Published Methods

| Method | Type | Ratio | SST-2 acc |
|---|---|---|---|
| ALBERT | Sharing | 3.5× | 89.4% |
| DistilBERT | Distill | 1.6× | 90.7% |
| **A3 (r=64)** | Low-rank | **{{D_RATIO}}×** | **{{D_ACC}}%** |
| **A3 (r=128)** | Low-rank | **{{D_RATIO128}}×** | **{{D_ACC128}}%** |

A3 at rank 64 achieves similar accuracy to ALBERT with similar compression 
but **no retraining from scratch**.

## 9.6 Deployment Considerations

### 9.6.1 Storage

A3's main benefit is storage. For a 100M-parameter model:

- Baseline: 200 MB FP16
- A3 (r=64): ~{{D_DISK}} MB
- A3 + INT8: ~{{D_DISK_INT8}} MB

### 9.6.2 Latency

A3's factorized layers are **~{{D_LATENCY_RATIO}}×** slower on GPU because 
they require two sequential matmuls. On CPU, they may be **faster** because 
smaller matrices fit in cache better.

### 9.6.3 Batching

A3 layers benefit more from larger batches because the smaller matrices 
amortize better.

## 9.7 Open Questions

1. **Interaction with quantization:** How does A3 + INT8 compare to INT8 alone?
2. **Cross-task transfer:** Do Fisher statistics transfer from SST-2 to other tasks?
3. **Optimal rank schedule:** Can a learned schedule outperform per-layer search?
4. **Fisher proxy accuracy:** How does $\mathbb{E}[x^2]$ compare to the true Fisher?

## 9.8 Broader Impact

### 9.8.1 Positive

- **Democratizes AI:** Smaller models → cheaper deployment.
- **Reduces carbon:** Less compute per inference.
- **Enables edge AI:** Private, offline inference.

### 9.8.2 Negative

- **May enable surveillance:** Cheaper deployment of harmful models.
- **Concentration risk:** Efficiency gains favor large labs.

### 9.8.3 Environmental

Total compute for A3 experiments: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.
