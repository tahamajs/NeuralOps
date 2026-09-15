<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Calibration-Free Compression

The method's defining feature is its **self-containedness**. No calibration 
data, no forward passes, no activation statistics. This makes it deployable 
in settings where:
- Data privacy restricts sharing
- The target task is unknown at compression time
- Fast compression is a hard requirement

### 9.1.2 Takeaway 2: Per-Layer Rank is Important

The energy threshold gives each projection its own rank, ranging from 
{{MIN_RANK_OBS}} to {{MAX_RANK_OBS}}. Fixed-rank at matched compression loses 
**{{FIXED_RANK_LOSS}}%** accuracy.

### 9.1.3 Takeaway 3: Adapters are Essential

Without residual adapters, accuracy drops by **{{NO_ADAPT_LOSS}}%**. With 
adapters, the drop is only **{{WITH_ADAPT_DROP}}%**. The adapter overhead is 
~{{ADAPTER_OVERHEAD}}% of the compressed base.

## 9.2 When to Use Adaptive SVD

### Recommended

- **Zero-data settings** (no calibration available)
- **Fast compression** (multi-task deployment)
- **Encoder-only models** (DistilBERT, BERT)
- **Multi-architecture pipelines** (T5, encoder-decoder)
- **Baseline for comparison** with calibration methods

### Not Recommended

- **Latency-critical** (two matmuls vs one)
- **Very tight VRAM** (adapter overhead)
- **Very large models** (>1B; SVD time dominates)
- **In-domain accuracy matters most** (A3/AASVD may be better)

## 9.3 Decision Flowchart

```
Have calibration data? ─── Yes ─── Consider A3/AASVD
  │ No
  ▼
Need fast compression? ─── Yes ─── Adaptive SVD
  │ No
  ▼
Latency critical? ─── Yes ─── Consider other methods
  │ No
  ▼
Adaptive SVD
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | Adaptive SVD | Change |
|---|---|---|---|
| Disk (FP16) | 134 MB | {{D_DISK}} MB | **{{D_DISK_RATIO}}×** |
| Params | 66.95M | {{D_PARAMS}}M | **{{D_RATIO}}×** |
| VRAM | 183 MB | {{D_VRAM}} MB | **{{D_VRAM_RATIO}}×** |
| Inference time | 1.90 s | {{D_TIME}} s | {{D_TIME_RATIO}}× |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| Compression time | — | {{D_TOTAL}} s | Fast |

**Verdict.** Adaptive SVD is a **strict win** for storage, VRAM, and 
compression speed.

## 9.5 Comparison with Published Methods

| Method | Type | Ratio | SST-2 acc | Calibration |
|---|---|---|---|---|
| ALBERT | Sharing | 3.5× | 89.4% | Retrain |
| DistilBERT | Distill | 1.6× | 90.7% | Retrain |
| A3 | Low-rank | {{A3_RATIO}}× | {{A3_D}} | Data |
| AASVD | Low-rank | {{AASVD_RATIO}}× | {{AASVD_D}} | Data |
| **Adaptive SVD** | Low-rank | **{{D_RATIO}}×** | **{{D_ACC}}** | **None** |

## 9.6 Deployment Considerations

### Storage
Same disk savings as A3/AASVD.

### Latency
Two matmuls + adapter = ~{{LATENCY_RATIO}}× slower than baseline. On CPU, 
may be faster due to cache effects.

### Compatibility
Works with HuggingFace models, LoRA frameworks, and quantization tools.

### Multi-Task
Since no calibration is required, one compressed model can serve multiple 
tasks without re-compression.

## 9.7 Open Questions

1. **Optimal τ per layer:** Could a learned per-layer $\tau$ beat uniform 
   $\tau = 0.92$?
2. **Combined with calibration:** How does Adaptive SVD compare to A3/AASVD 
   when both have calibration data?
3. **Interaction with head pruning:** Can both be combined?
4. **Cross-lingual transfer:** Does energy-based rank transfer across languages?
5. **Theoretical bounds:** Can we bound downstream accuracy loss?

## 9.8 Broader Impact

### Positive
- **Democratizes compression:** No calibration data → broader access
- **Faster iteration:** Compress in seconds, not minutes
- **Multi-architecture support:** Works on T5, encoder-decoder, GQA

### Negative
- **Dual-use:** Efficiency benefits malicious and beneficial applications
- **Concentration:** Large labs may benefit most

### Environmental
Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.
