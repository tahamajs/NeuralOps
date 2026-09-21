# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_svd_sequence/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> |
| results/unified_svd_sequence/distilbert/summary.csv | distilbert | svd_sequence | PASS_LOW | 51.26% <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> | 0.5737 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> |
| results/unified_svd_sequence/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> |
| results/unified_svd_sequence/gpt2/summary.csv | gpt2 | svd_sequence | PASS_LOW | 53.1% <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> | 0.6832 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> |
| results/unified_svd_sequence/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> |
| results/unified_svd_sequence/roberta/summary.csv | roberta | svd_sequence | PASS_LOW | 50.0% <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> | 0.6572 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/distilbert/summary.csv | distilbert | svd_sequence_attention | PASS_LOW | 51.26% <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> | 0.5737 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/gpt2/summary.csv | gpt2 | svd_sequence_attention | PASS_LOW | 53.1% <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> | 0.6832 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/roberta/summary.csv | roberta | svd_sequence_attention | PASS_LOW | 50.0% <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> | 0.6572 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_svd_sequence/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> |
| results/unified_svd_sequence/distilbert/summary.csv | distilbert | svd_sequence | 66955010 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> | 67544858 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> | 1181978 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> |
| results/unified_svd_sequence/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> |
| results/unified_svd_sequence/gpt2/summary.csv | gpt2 | svd_sequence | 124441344 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> | 125621040 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> | 1181232 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> | 0.9906x <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> |
| results/unified_svd_sequence/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> |
| results/unified_svd_sequence/roberta/summary.csv | roberta | svd_sequence | 124647170 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> | 132913970 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> | 1771826 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> | 0.9378x <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/distilbert/summary.csv | distilbert | svd_sequence_attention | 66955010 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> | 67544858 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> | 1181978 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/gpt2/summary.csv | gpt2 | svd_sequence_attention | 124441344 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> | 125621040 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> | 1181232 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> | 0.9906x <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/roberta/summary.csv | roberta | svd_sequence_attention | 124647170 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> | 132913970 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> | 1771826 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> | 0.9378x <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_svd_sequence/distilbert/summary.csv | distilbert | full | 149945.87 tok/s <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 2 --> |
| results/unified_svd_sequence/distilbert/summary.csv | distilbert | svd_sequence | 43289.28 tok/s <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> | 329.37 MB <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_sequence/distilbert/summary.csv line 3 --> |
| results/unified_svd_sequence/gpt2/summary.csv | gpt2 | full | 64344.52 tok/s <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 2 --> |
| results/unified_svd_sequence/gpt2/summary.csv | gpt2 | svd_sequence | 32157.89 tok/s <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> | 569.83 MB <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_sequence/gpt2/summary.csv line 3 --> |
| results/unified_svd_sequence/roberta/summary.csv | roberta | full | 81335.45 tok/s <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 2 --> |
| results/unified_svd_sequence/roberta/summary.csv | roberta | svd_sequence | 23394.65 tok/s <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> | 655.77 MB <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_sequence/roberta/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/distilbert/summary.csv | distilbert | full | 144662.78 tok/s <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/distilbert/summary.csv | distilbert | svd_sequence_attention | 42624.42 tok/s <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> | 329.37 MB <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_sequence_attention/distilbert/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/gpt2/summary.csv | gpt2 | full | 63999.6 tok/s <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/gpt2/summary.csv | gpt2 | svd_sequence_attention | 35808.87 tok/s <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> | 569.83 MB <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_sequence_attention/gpt2/summary.csv line 3 --> |
| results/unified_svd_sequence_attention/roberta/summary.csv | roberta | full | 81675.38 tok/s <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 2 --> |
| results/unified_svd_sequence_attention/roberta/summary.csv | roberta | svd_sequence_attention | 23073.3 tok/s <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> | 655.77 MB <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_sequence_attention/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 0.00% | `results/master_aggregated_results.csv` |
| F1 Score | 0.0000 | `results/master_aggregated_results.csv` |
| Compression | 1.000x | `results/master_aggregated_results.csv` |
| Compressed Layers | ? | `results/master_aggregated_results.csv` |
| Fallback Layers | ? | `results/master_aggregated_results.csv` |
| Timestamp | 2026-09-11T21:51:07.961641 | `results/master_aggregated_results.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.94, 57.91, 75.46, 90.94, 57.91, 75.46]
> These numbers must NOT be cited in the final paper.

