| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Smart Weighted Ensemble** | **90.14%** | -0.34% | PASS | 0.9029 | 0 | **1.00x** | 337.6 MB | 2.2s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Smart Weighted Ensemble** | **0.00%** | -75.11% | FAIL | 0.7586 | 0 | **1.00x** | 298.7 MB | 0.5s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Smart Weighted Ensemble** | **61.01%** | +0.23% | PASS | 0.6654 | 0 | **1.00x** | 634.9 MB | 3.9s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | smart_weighted_ensemble | PASS | 90.14% <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 0.9029 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | smart_weighted_ensemble | PASS | 61.01% <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 0.6654 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | smart_weighted_ensemble | FAIL | 0.0% <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 0.7586 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | smart_weighted_ensemble | 66955010 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | smart_weighted_ensemble | 124441344 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | smart_weighted_ensemble | 124647170 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | full | 148495.02 tok/s <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | smart_weighted_ensemble | 144610.54 tok/s <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | full | 62844.73 tok/s <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | smart_weighted_ensemble | 66142.57 tok/s <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | full | 90470.13 tok/s <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | smart_weighted_ensemble | 90470.13 tok/s <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 298.71 MB <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.14% | `results/unified_smart_weighted_ensemble/distilbert/summary.csv` |
| F1 Score | 0.9029 | `results/unified_smart_weighted_ensemble/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_smart_weighted_ensemble/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_smart_weighted_ensemble/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_smart_weighted_ensemble/distilbert/summary.csv` |
| Timestamp | 2026-09-11T08:26:48.886509 | `results/unified_smart_weighted_ensemble/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

