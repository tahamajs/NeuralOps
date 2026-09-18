# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_minimal_svd_v2/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/distilbert/summary.csv | distilbert | minimal_svd_v2 | PASS | 88.19% <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> | 0.8891 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> |
| results/unified_minimal_svd_v2/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/gpt2/summary.csv | gpt2 | minimal_svd_v2 | PASS | 59.86% <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> | 0.7107 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> |
| results/unified_minimal_svd_v2/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/roberta/summary.csv | roberta | minimal_svd_v2 | PASS | 64.56% <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> | 0.7193 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_minimal_svd_v2/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/distilbert/summary.csv | distilbert | minimal_svd_v2 | 66955010 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> | 58107656 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> | 589830 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> | 1.1523x <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> |
| results/unified_minimal_svd_v2/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/gpt2/summary.csv | gpt2 | minimal_svd_v2 | 124441344 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> | 106746636 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> | 1179660 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> | 1.1658x <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> |
| results/unified_minimal_svd_v2/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/roberta/summary.csv | roberta | minimal_svd_v2 | 124647170 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> | 114039566 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> | 1179660 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> | 1.093x <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_minimal_svd_v2/distilbert/summary.csv | distilbert | full | 58001.39 tok/s <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/distilbert/summary.csv | distilbert | minimal_svd_v2 | 25832.74 tok/s <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> | 173.28 MB <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_minimal_svd_v2/distilbert/summary.csv line 3 --> |
| results/unified_minimal_svd_v2/gpt2/summary.csv | gpt2 | full | 27623.29 tok/s <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/gpt2/summary.csv | gpt2 | minimal_svd_v2 | 11161.39 tok/s <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> | 495.67 MB <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_minimal_svd_v2/gpt2/summary.csv line 3 --> |
| results/unified_minimal_svd_v2/roberta/summary.csv | roberta | full | 27376.47 tok/s <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 2 --> |
| results/unified_minimal_svd_v2/roberta/summary.csv | roberta | minimal_svd_v2 | 13146.62 tok/s <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> | 579.51 MB <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_minimal_svd_v2/roberta/summary.csv line 3 --> |
