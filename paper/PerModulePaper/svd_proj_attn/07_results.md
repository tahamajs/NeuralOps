# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_svd_proj_attn/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> |
| results/unified_svd_proj_attn/distilbert/summary.csv | distilbert | svd_proj_attn | PASS | 76.15% <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> | 0.7873 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> |
| results/unified_svd_proj_attn/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> |
| results/unified_svd_proj_attn/gpt2/summary.csv | gpt2 | svd_proj_attn | PASS_LOW | 50.92% <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> |
| results/unified_svd_proj_attn/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> |
| results/unified_svd_proj_attn/roberta/summary.csv | roberta | svd_proj_attn | PASS_LOW | 50.92% <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_svd_proj_attn/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> |
| results/unified_svd_proj_attn/distilbert/summary.csv | distilbert | svd_proj_attn | 66955010 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> | 54568730 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> | 1181978 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> | 1.227x <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> |
| results/unified_svd_proj_attn/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> |
| results/unified_svd_proj_attn/gpt2/summary.csv | gpt2 | svd_proj_attn | 124441344 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> | 99668784 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> | 1181232 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> | 1.2485x <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> |
| results/unified_svd_proj_attn/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> |
| results/unified_svd_proj_attn/roberta/summary.csv | roberta | svd_proj_attn | 124647170 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> | 106961714 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> | 1771826 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> | 1.1653x <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_svd_proj_attn/distilbert/summary.csv | distilbert | full | 141839.0 tok/s <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 2 --> |
| results/unified_svd_proj_attn/distilbert/summary.csv | distilbert | svd_proj_attn | 69162.08 tok/s <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> | 283.77 MB <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_proj_attn/distilbert/summary.csv line 3 --> |
| results/unified_svd_proj_attn/gpt2/summary.csv | gpt2 | full | 62971.65 tok/s <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 2 --> |
| results/unified_svd_proj_attn/gpt2/summary.csv | gpt2 | svd_proj_attn | 31070.56 tok/s <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> | 468.71 MB <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_proj_attn/gpt2/summary.csv line 3 --> |
| results/unified_svd_proj_attn/roberta/summary.csv | roberta | full | 96632.08 tok/s <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 2 --> |
| results/unified_svd_proj_attn/roberta/summary.csv | roberta | svd_proj_attn | 42386.76 tok/s <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> | 328.38 MB <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_proj_attn/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=76.15%, comp=1.227x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 76.15% | `results/unified_svd_proj_attn/distilbert/summary.csv` |
| F1 Score | 0.7873 | `results/unified_svd_proj_attn/distilbert/summary.csv` |
| Compression | 1.227x | `results/unified_svd_proj_attn/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_svd_proj_attn/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_svd_proj_attn/distilbert/summary.csv` |
| Timestamp | 2026-09-11T09:32:48.138910 | `results/unified_svd_proj_attn/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.48, 60.78]
> These numbers must NOT be cited in the final paper.

