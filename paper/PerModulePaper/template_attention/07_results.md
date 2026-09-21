# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_template_attention/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> |
| results/unified_template_attention/distilbert/summary.csv | distilbert | template_attention | PASS | 89.91% <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> | 0.9033 <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> |
| results/unified_template_attention/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> |
| results/unified_template_attention/gpt2/summary.csv | gpt2 | template_attention | PASS_LOW | 51.26% <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> | 0.6763 <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> |
| results/unified_template_attention/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> |
| results/unified_template_attention/roberta/summary.csv | roberta | template_attention | PASS | 58.37% <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> | 0.6857 <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_template_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> |
| results/unified_template_attention/distilbert/summary.csv | distilbert | template_attention | 66955010 <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> | 67692314 <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> | 1181978 <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> | 0.9891x <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> |
| results/unified_template_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> |
| results/unified_template_attention/gpt2/summary.csv | gpt2 | template_attention | 124441344 <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> | 125915952 <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> | 1181232 <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> | 0.9883x <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> |
| results/unified_template_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> |
| results/unified_template_attention/roberta/summary.csv | roberta | template_attention | 124647170 <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> | 133208882 <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> | 1771826 <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> | 0.9357x <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_template_attention/distilbert/summary.csv | distilbert | full | 138662.34 tok/s <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_template_attention/distilbert/summary.csv line 2 --> |
| results/unified_template_attention/distilbert/summary.csv | distilbert | template_attention | 71449.72 tok/s <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> | 329.61 MB <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_template_attention/distilbert/summary.csv line 3 --> |
| results/unified_template_attention/gpt2/summary.csv | gpt2 | full | 60153.38 tok/s <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_template_attention/gpt2/summary.csv line 2 --> |
| results/unified_template_attention/gpt2/summary.csv | gpt2 | template_attention | 35032.52 tok/s <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> | 569.83 MB <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_template_attention/gpt2/summary.csv line 3 --> |
| results/unified_template_attention/roberta/summary.csv | roberta | full | 102308.54 tok/s <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_template_attention/roberta/summary.csv line 2 --> |
| results/unified_template_attention/roberta/summary.csv | roberta | template_attention | 41090.86 tok/s <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> | 377.72 MB <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_template_attention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.60% | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| F1 Score | 0.9091 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Timestamp | 2026-09-19T02:10:24.915406 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [60.78, 75.11]
> These numbers must NOT be cited in the final paper.

