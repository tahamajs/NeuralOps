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
64 to 150. Fixed-rank at matched compression loses 
**1.5%** accuracy.

### 9.1.3 Takeaway 3: Adapters are Essential

Without residual adapters, accuracy drops by **8%**. With 
adapters, the drop is only **2.7%**. The adapter overhead is 
~12% of the compressed base.

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
| Disk (FP16) | 134 MB | 45 MB | **3.0×** |
| Params | 66.95M | 22000000M | **3.0×** |
| VRAM | 183 MB | 130 MB | **2.1\times×** |
| Inference time | 1.90 s | 2.3 s | 1.2× |
| Accuracy | 90.71% | 88.0% | -2.7% |
| Compression time | — | 15 s | Fast |

**Verdict.** Adaptive SVD is a **strict win** for storage, VRAM, and 
compression speed.

## 9.5 Comparison with Published Methods

| Method | Type | Ratio | SST-2 acc | Calibration |
|---|---|---|---|---|
| ALBERT | Sharing | 3.5× | 89.4% | Retrain |
| DistilBERT | Distill | 1.6× | 90.7% | Retrain |
| A3 | Low-rank | 2.1\times× | 85.5 | Data |
| AASVD | Low-rank | 2.1\times× | 85.0 | Data |
| **Adaptive SVD** | Low-rank | **3.0×** | **88.0** | **None** |

## 9.6 Deployment Considerations

### Storage
Same disk savings as A3/AASVD.

### Latency
Two matmuls + adapter = ~1.2× slower than baseline. On CPU, 
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
Total compute: ~10 GPU-hours, ~12.4 kg CO₂eq.
