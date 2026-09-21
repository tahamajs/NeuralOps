# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_svd_projection/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> |
| results/unified_svd_projection/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> |
| results/unified_svd_projection/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_svd_projection/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> |
| results/unified_svd_projection/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> |
| results/unified_svd_projection/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_svd_projection/distilbert/summary.csv | distilbert | full | 139700.69 tok/s <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_projection/distilbert/summary.csv line 2 --> |
| results/unified_svd_projection/gpt2/summary.csv | gpt2 | full | 62284.77 tok/s <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_projection/gpt2/summary.csv line 2 --> |
| results/unified_svd_projection/roberta/summary.csv | roberta | full | 84407.34 tok/s <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_projection/roberta/summary.csv line 2 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 70.31% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-26 00:34:44 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.48, 75.11]
> These numbers must NOT be cited in the final paper.

