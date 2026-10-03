# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_svd/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> |
| results/unified_svd/distilbert/summary.csv | distilbert | svd | PASS_LOW | 49.31% <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> | 0.0178 <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> |
| results/unified_svd/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> |
| results/unified_svd/gpt2/summary.csv | gpt2 | svd | PASS_NO_COMPRESSION | 49.08% <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> |
| results/unified_svd/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_svd/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_svd/roberta/summary.csv line 2 --> |
| results/unified_svd/roberta/summary.csv | roberta | svd | PASS_NO_COMPRESSION | 74.08% <!-- source: results/unified_svd/roberta/summary.csv line 3 --> | 0.7675 <!-- source: results/unified_svd/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_svd/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> |
| results/unified_svd/distilbert/summary.csv | distilbert | svd | 66955010 <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> | 67544834 <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> | 589824 <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> |
| results/unified_svd/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> |
| results/unified_svd/gpt2/summary.csv | gpt2 | svd | 124441344 <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> | 125620992 <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> | 1179648 <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> |
| results/unified_svd/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_svd/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_svd/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_svd/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd/roberta/summary.csv line 2 --> |
| results/unified_svd/roberta/summary.csv | roberta | svd | 124647170 <!-- source: results/unified_svd/roberta/summary.csv line 3 --> | 132913922 <!-- source: results/unified_svd/roberta/summary.csv line 3 --> | 1179648 <!-- source: results/unified_svd/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_svd/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_svd/distilbert/summary.csv | distilbert | full | 25596.23 tok/s <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> | 388.55 MB <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd/distilbert/summary.csv line 2 --> |
| results/unified_svd/distilbert/summary.csv | distilbert | svd | 8596.59 tok/s <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> | 317.64 MB <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd/distilbert/summary.csv line 3 --> |
| results/unified_svd/gpt2/summary.csv | gpt2 | full | 27624.42 tok/s <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> | 792.54 MB <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd/gpt2/summary.csv line 2 --> |
| results/unified_svd/gpt2/summary.csv | gpt2 | svd | 12404.45 tok/s <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> | 590.39 MB <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd/gpt2/summary.csv line 3 --> |
| results/unified_svd/roberta/summary.csv | roberta | full | 28095.04 tok/s <!-- source: results/unified_svd/roberta/summary.csv line 2 --> | 711.69 MB <!-- source: results/unified_svd/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd/roberta/summary.csv line 2 --> |
| results/unified_svd/roberta/summary.csv | roberta | svd | 12718.31 tok/s <!-- source: results/unified_svd/roberta/summary.csv line 3 --> | 680.69 MB <!-- source: results/unified_svd/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 75.00% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-25 23:52:49 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 79.36, 78.56]
> These numbers must NOT be cited in the final paper.

