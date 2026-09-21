# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_wgqa/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> |
| results/unified_wgqa/distilbert/summary.csv | distilbert | wgqa | PASS | 90.6% <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> | 0.9089 <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> |
| results/unified_wgqa/gpt2/summary.csv | gpt2 | full | PASS | 87.84% <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> | 0.8776 <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> |
| results/unified_wgqa/gpt2/summary.csv | gpt2 | wgqa | PASS | 87.16% <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> | 0.879 <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> |
| results/unified_wgqa/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> |
| results/unified_wgqa/roberta/summary.csv | roberta | wgqa | PASS_LOW | 52.64% <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> | 0.1342 <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_wgqa/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> |
| results/unified_wgqa/distilbert/summary.csv | distilbert | wgqa | 66955010 <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> | 67545002 <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> | 589992 <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> |
| results/unified_wgqa/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> |
| results/unified_wgqa/gpt2/summary.csv | gpt2 | wgqa | 126800640 <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> | 127095888 <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> | 2654544 <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> | 0.9977x <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> |
| results/unified_wgqa/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> |
| results/unified_wgqa/roberta/summary.csv | roberta | wgqa | 124647170 <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> | 132914258 <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> | 1179984 <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> | 0.9378x <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_wgqa/distilbert/summary.csv | distilbert | full | 52577.79 tok/s <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_wgqa/distilbert/summary.csv line 2 --> |
| results/unified_wgqa/distilbert/summary.csv | distilbert | wgqa | 24323.59 tok/s <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> | 326.42 MB <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_wgqa/distilbert/summary.csv line 3 --> |
| results/unified_wgqa/gpt2/summary.csv | gpt2 | full | 26531.63 tok/s <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_wgqa/gpt2/summary.csv line 2 --> |
| results/unified_wgqa/gpt2/summary.csv | gpt2 | wgqa | 8935.39 tok/s <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> | 343.63 MB <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_wgqa/gpt2/summary.csv line 3 --> |
| results/unified_wgqa/roberta/summary.csv | roberta | full | 26639.02 tok/s <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_wgqa/roberta/summary.csv line 2 --> |
| results/unified_wgqa/roberta/summary.csv | roberta | wgqa | 13334.1 tok/s <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> | 655.06 MB <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_wgqa/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.60% | `results/unified_sweep/distilbert/summary.csv` |
| F1 Score | 0.9089 | `results/unified_sweep/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary.csv` |
| Timestamp | 2026-09-19T02:15:17.185790 | `results/unified_sweep/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.46]
> These numbers must NOT be cited in the final paper.

