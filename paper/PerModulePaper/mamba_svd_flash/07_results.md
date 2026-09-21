# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_mamba_svd_flash/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/distilbert/summary.csv | distilbert | mamba_svd_flash | FAIL | 0.0% <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> |
| results/unified_mamba_svd_flash/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/gpt2/summary.csv | gpt2 | mamba_svd_flash | FAIL | 0.0% <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> | 0.6354 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> |
| results/unified_mamba_svd_flash/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/roberta/summary.csv | roberta | mamba_svd_flash | FAIL | 0.0% <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> | 0.7586 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_mamba_svd_flash/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/distilbert/summary.csv | distilbert | mamba_svd_flash | 66955010 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> |
| results/unified_mamba_svd_flash/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/gpt2/summary.csv | gpt2 | mamba_svd_flash | 124441344 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> |
| results/unified_mamba_svd_flash/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/roberta/summary.csv | roberta | mamba_svd_flash | 124647170 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_mamba_svd_flash/distilbert/summary.csv | distilbert | full | 58298.82 tok/s <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/distilbert/summary.csv | distilbert | mamba_svd_flash | 58298.82 tok/s <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> | 182.94 MB <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> |
| results/unified_mamba_svd_flash/gpt2/summary.csv | gpt2 | full | 27154.27 tok/s <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/gpt2/summary.csv | gpt2 | mamba_svd_flash | 64043.03 tok/s <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> |
| results/unified_mamba_svd_flash/roberta/summary.csv | roberta | full | 26868.5 tok/s <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 2 --> |
| results/unified_mamba_svd_flash/roberta/summary.csv | roberta | mamba_svd_flash | 92803.91 tok/s <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> | 298.71 MB <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 62.50% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-25 23:53:33 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 79.36, 78.56]
> These numbers must NOT be cited in the final paper.

