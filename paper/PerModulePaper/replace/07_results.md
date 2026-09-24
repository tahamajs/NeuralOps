| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Replace** | **0.00%** | -90.48% | FAIL | 0.9077 | 0 | **1.00x** | 389.5 MB | 0.3s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Replace** | **0.00%** | -75.11% | FAIL | 0.7586 | 0 | **1.00x** | 298.7 MB | 0.5s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Replace** | **0.00%** | -60.78% | FAIL | 0.6354 | 0 | **1.00x** | 793.2 MB | 0.5s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_replace/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> |
| results/unified_replace/distilbert/summary.csv | distilbert | replace | FAIL | 0.0% <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> | 0.9077 <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> |
| results/unified_replace/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> |
| results/unified_replace/gpt2/summary.csv | gpt2 | replace | FAIL | 0.0% <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> | 0.6354 <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> |
| results/unified_replace/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_replace/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_replace/roberta/summary.csv line 2 --> |
| results/unified_replace/roberta/summary.csv | roberta | replace | FAIL | 0.0% <!-- source: results/unified_replace/roberta/summary.csv line 3 --> | 0.7586 <!-- source: results/unified_replace/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_replace/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> |
| results/unified_replace/distilbert/summary.csv | distilbert | replace | 66955010 <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> |
| results/unified_replace/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> |
| results/unified_replace/gpt2/summary.csv | gpt2 | replace | 124441344 <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> |
| results/unified_replace/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_replace/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_replace/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_replace/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_replace/roberta/summary.csv line 2 --> |
| results/unified_replace/roberta/summary.csv | roberta | replace | 124647170 <!-- source: results/unified_replace/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_replace/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_replace/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_replace/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_replace/distilbert/summary.csv | distilbert | full | 144873.64 tok/s <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_replace/distilbert/summary.csv line 2 --> |
| results/unified_replace/distilbert/summary.csv | distilbert | replace | 144873.64 tok/s <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> | 389.51 MB <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_replace/distilbert/summary.csv line 3 --> |
| results/unified_replace/gpt2/summary.csv | gpt2 | full | 63288.7 tok/s <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_replace/gpt2/summary.csv line 2 --> |
| results/unified_replace/gpt2/summary.csv | gpt2 | replace | 63288.7 tok/s <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_replace/gpt2/summary.csv line 3 --> |
| results/unified_replace/roberta/summary.csv | roberta | full | 86937.12 tok/s <!-- source: results/unified_replace/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_replace/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_replace/roberta/summary.csv line 2 --> |
| results/unified_replace/roberta/summary.csv | roberta | replace | 86937.12 tok/s <!-- source: results/unified_replace/roberta/summary.csv line 3 --> | 298.71 MB <!-- source: results/unified_replace/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_replace/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 92.60% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9223 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T18:22:50.707941 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.48, 60.78, 75.11]
> These numbers must NOT be cited in the final paper.

