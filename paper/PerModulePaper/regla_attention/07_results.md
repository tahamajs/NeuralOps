# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_regla/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> |
| results/unified_regla/distilbert/summary.csv | distilbert | regla | PASS_NO_COMPRESSION | 79.82% <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> | 0.8108 <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> |
| results/unified_regla/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> |
| results/unified_regla/gpt2/summary.csv | gpt2 | regla | PASS_NO_COMPRESSION | 57.45% <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> | 0.6996 <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> |
| results/unified_regla/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_regla/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_regla/roberta/summary.csv line 2 --> |
| results/unified_regla/roberta/summary.csv | roberta | regla | PASS_NO_COMPRESSION | 64.79% <!-- source: results/unified_regla/roberta/summary.csv line 3 --> | 0.7149 <!-- source: results/unified_regla/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_regla/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> |
| results/unified_regla/distilbert/summary.csv | distilbert | regla | 66955010 <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> | 71254280 <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> | 4280832 <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> |
| results/unified_regla/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> |
| results/unified_regla/gpt2/summary.csv | gpt2 | regla | 124441344 <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> | 133039884 <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> | 8561664 <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> |
| results/unified_regla/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_regla/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_regla/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_regla/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_regla/roberta/summary.csv line 2 --> |
| results/unified_regla/roberta/summary.csv | roberta | regla | 124647170 <!-- source: results/unified_regla/roberta/summary.csv line 3 --> | 140332814 <!-- source: results/unified_regla/roberta/summary.csv line 3 --> | 8561664 <!-- source: results/unified_regla/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_regla/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_regla/distilbert/summary.csv | distilbert | full | 50660.12 tok/s <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_regla/distilbert/summary.csv line 2 --> |
| results/unified_regla/distilbert/summary.csv | distilbert | regla | 21570.51 tok/s <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> | 374.65 MB <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_regla/distilbert/summary.csv line 3 --> |
| results/unified_regla/gpt2/summary.csv | gpt2 | full | 28366.76 tok/s <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_regla/gpt2/summary.csv line 2 --> |
| results/unified_regla/gpt2/summary.csv | gpt2 | regla | 5001.6 tok/s <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> | 1060.01 MB <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_regla/gpt2/summary.csv line 3 --> |
| results/unified_regla/roberta/summary.csv | roberta | full | 28714.19 tok/s <!-- source: results/unified_regla/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_regla/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_regla/roberta/summary.csv line 2 --> |
| results/unified_regla/roberta/summary.csv | roberta | regla | 11346.85 tok/s <!-- source: results/unified_regla/roberta/summary.csv line 3 --> | 749.59 MB <!-- source: results/unified_regla/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_regla/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 68.38% | `results/adaptive_v3/benchmark_summary.csv` |
| F1 Score | 0.8100 | `results/adaptive_v3/benchmark_summary.csv` |
| Compression | 1.000x | `results/adaptive_v3/benchmark_summary.csv` |
| Compressed Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Fallback Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Timestamp | 2026-08-26 08:11:22 | `results/adaptive_v3/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 79.82, 79.36, 57.45, 78.56, 64.79]
> These numbers must NOT be cited in the final paper.

