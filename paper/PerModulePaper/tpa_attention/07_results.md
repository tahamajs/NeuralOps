# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_tpa/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> |
<!-- FABRICATED: | results/unified_tpa/distilbert/summary.csv | distilbert | tpa | PASS_NO_COMPRESSION | 75.0% <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> | 0.6997 <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> | -->
| results/unified_tpa/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> |
| results/unified_tpa/gpt2/summary.csv | gpt2 | tpa | PASS_LOW | 49.08% <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> |
| results/unified_tpa/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> |
| results/unified_tpa/roberta/summary.csv | roberta | tpa | PASS | 59.75% <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> | 0.623 <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_tpa/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> |
| results/unified_tpa/distilbert/summary.csv | distilbert | tpa | 66955010 <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> | 95650562 <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> | 1179648 <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> |
| results/unified_tpa/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> |
| results/unified_tpa/gpt2/summary.csv | gpt2 | tpa | 124441344 <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> | 115059456 <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> | 1253376 <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> | 1.0815x <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> |
| results/unified_tpa/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> |
| results/unified_tpa/roberta/summary.csv | roberta | tpa | 124647170 <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> | 176542538 <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> | 2361674 <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> | 0.706x <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_tpa/distilbert/summary.csv | distilbert | full | 6751.04 tok/s <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> | 379.42 MB <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tpa/distilbert/summary.csv line 2 --> |
| results/unified_tpa/distilbert/summary.csv | distilbert | tpa | 2620.26 tok/s <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> | 605.71 MB <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tpa/distilbert/summary.csv line 3 --> |
| results/unified_tpa/gpt2/summary.csv | gpt2 | full | 13704.77 tok/s <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> | 728.04 MB <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tpa/gpt2/summary.csv line 2 --> |
| results/unified_tpa/gpt2/summary.csv | gpt2 | tpa | 4692.26 tok/s <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> | 500.77 MB <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tpa/gpt2/summary.csv line 3 --> |
| results/unified_tpa/roberta/summary.csv | roberta | full | 45523.74 tok/s <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> | 698.89 MB <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tpa/roberta/summary.csv line 2 --> |
| results/unified_tpa/roberta/summary.csv | roberta | tpa | 2203.51 tok/s <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> | 2738.19 MB <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_tpa/roberta/summary.csv line 3 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `NO_DATA`  
**Detail**: No CSV rows found

*No valid CSV data found for this module.*


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 75.0, 79.36, 75.46, 59.75]
> These numbers must NOT be cited in the final paper.

