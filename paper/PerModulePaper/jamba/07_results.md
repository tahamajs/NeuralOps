# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_jamba/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> |
| results/unified_jamba/distilbert/summary.csv | distilbert | jamba | PASS | 87.16% <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> | 0.8753 <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> |
| results/unified_jamba/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> |
| results/unified_jamba/gpt2/summary.csv | gpt2 | jamba | PASS_LOW | 52.29% <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> | 0.6745 <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> |
| results/unified_jamba/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> |
| results/unified_jamba/roberta/summary.csv | roberta | jamba | PASS_LOW | 50.92% <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> | 0.6733 <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_jamba/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> |
| results/unified_jamba/distilbert/summary.csv | distilbert | jamba | 66955010 <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> | 160322336 <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> | 1698080 <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> | 0.4176x <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> |
| results/unified_jamba/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> |
| results/unified_jamba/gpt2/summary.csv | gpt2 | jamba | 124441344 <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> | 332437308 <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> | 2213436 <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> | 0.3743x <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> |
| results/unified_jamba/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> |
| results/unified_jamba/roberta/summary.csv | roberta | jamba | 124647170 <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> | 318468926 <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> | 2214206 <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> | 0.3914x <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_jamba/distilbert/summary.csv | distilbert | full | 167898.9 tok/s <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_jamba/distilbert/summary.csv line 2 --> |
| results/unified_jamba/distilbert/summary.csv | distilbert | jamba | 25763.56 tok/s <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> | 716.63 MB <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> |
| results/unified_jamba/gpt2/summary.csv | gpt2 | full | 70977.91 tok/s <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_jamba/gpt2/summary.csv line 2 --> |
| results/unified_jamba/gpt2/summary.csv | gpt2 | jamba | 15321.66 tok/s <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> | 1365.61 MB <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> |
| results/unified_jamba/roberta/summary.csv | roberta | full | 89113.64 tok/s <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_jamba/roberta/summary.csv line 2 --> |
| results/unified_jamba/roberta/summary.csv | roberta | jamba | 26521.15 tok/s <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> | 1321.02 MB <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 88.99% | `results/distilbert/summary.csv` |
| F1 Score | 0.0000 | `results/distilbert/summary.csv` |
| Compression | 1.000x | `results/distilbert/summary.csv` |
| Compressed Layers | ? | `results/distilbert/summary.csv` |
| Fallback Layers | ? | `results/distilbert/summary.csv` |
| Timestamp | 2026-09-04T07:41:29.484167 | `results/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.46]
> These numbers must NOT be cited in the final paper.

