<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Reproducibility

MLA reproduces faithfully on DistilBERT, RoBERTa, and GPT-2. The main 
challenges are architectural portability, not algorithmic.

### 9.1.2 Takeaway 2: Real KV Compression

MLA reduces KV-cache size by **{{D_KV_RATIO}}×–{{G_KV_RATIO}}×**. This is 
unique among the compression methods we compared.

### 9.1.3 Takeaway 3: Complementary to Parameter Compression

MLA is orthogonal to parameter-compression methods (MinimalSVD, SVD-LLM V2). 
They can be combined.

## 9.2 When to Use MLA

### Recommended

- **Long-context inference** (KV cache is the bottleneck)
- **High batch size** (KV cache scales with B × L)
- **Models with concentrated K/V spectra**
- **Combined with quantization**

### Not Recommended

- **Memory-bound by parameters** (not KV cache)
- **Very small models** (<10M params)
- **Extremely small latent dimension** ($c < 32$)

## 9.3 Decision Flowchart

```
Inference bottleneck? ─── KV cache ─── Use MLA
  │
  Parameters ─── Use MinimalSVD / SVD-LLM V2
  │
  Compute ─── Consider other approaches

  │
  ▼
Have long context? ─── Yes ─── MLA gives more savings
  │ No
  ▼
MLA still viable for high-batch workloads
```

## 9.4 Cost-Benefit Analysis

For DistilBERT inference, batch 8, seq 128:

| Metric | Baseline | MLA | Change |
|---|---|---|---|
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| KV cache (MB) | {{D_KV_STD}} | {{D_KV_MLA}} | **{{D_KV_RATIO}}×** |
| Total VRAM (MB) | 183 | {{D_VRAM}} | {{D_VRAM_RATIO}}× |
| Latency | 1.90 s | {{D_TIME}} s | {{D_TIME_RATIO}}× |

For long context, seq 2048:

| Metric | Baseline | MLA | Change |
|---|---|---|---|
| KV cache (MB) | {{D_KV2048_STD}} | {{D_KV2048_MLA}} | **{{D_KV_RATIO}}×** |

**Verdict.** MLA is a **strict win** for KV-cache-bound workloads.

## 9.5 Comparison with Published Methods

| Method | KV reduction | Params | Accuracy |
|---|---|---|---|
| Standard MHA | 1× | 100% | Baseline |
| MQA | 12× | 100% | Lower |
| GQA (2 groups) | 6× | 100% | Similar |
| H2O | Variable | 100% | Lossy |
| **MLA (ours)** | **{{D_KV_RATIO}}×** | 100% | {{D_ACC}}% |

## 9.6 Deployment Considerations

### Storage
MLA does not significantly reduce storage (only K/V down-projection).

### Latency
MLA adds an extra down/up projection step but saves memory bandwidth 
for large batches.

### Batching
Benefits scale with batch size due to KV-cache size scaling.

## 9.7 Open Questions

1. **Interaction with quantization:** MLA + INT8 KV?
2. **Combined with parameter compression:** MLA + MinimalSVD?
3. **Optimal latent dimension:** can we choose $c$ automatically?
4. **Cross-task generalization:** does MLA work on generation?

## 9.8 Broader Impact

### Positive
- Enables longer context inference
- Reduces memory footprint
- Democratizes long-context AI

### Negative
- Dual-use concerns
- Concentration risk

### Environmental
Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.

---