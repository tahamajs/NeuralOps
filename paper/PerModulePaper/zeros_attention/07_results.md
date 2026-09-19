# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_zeros/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> |
| results/unified_zeros/distilbert/summary.csv | distilbert | zeros_attention | PASS_LOW | 51.83% <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> | 0.6789 <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> |
| results/unified_zeros/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> |
| results/unified_zeros/gpt2/summary.csv | gpt2 | zeros_attention | PASS | 59.75% <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> | 0.7023 <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> |
| results/unified_zeros/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> |
| results/unified_zeros/roberta/summary.csv | roberta | zeros_attention | PASS_LOW | 50.92% <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_zeros/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> |
| results/unified_zeros/distilbert/summary.csv | distilbert | zeros_attention | 66955010 <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> | 67544858 <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> | 1181978 <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> |
| results/unified_zeros/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> |
| results/unified_zeros/gpt2/summary.csv | gpt2 | zeros_attention | 124441344 <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> | 125621040 <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> | 1181232 <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> | 0.9906x <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> |
| results/unified_zeros/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> |
| results/unified_zeros/roberta/summary.csv | roberta | zeros_attention | 124647170 <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> | 132913970 <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> | 1771826 <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> | 0.9378x <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_zeros/distilbert/summary.csv | distilbert | full | 146466.62 tok/s <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_zeros/distilbert/summary.csv line 2 --> |
| results/unified_zeros/distilbert/summary.csv | distilbert | zeros_attention | 74818.79 tok/s <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> | 330.51 MB <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_zeros/distilbert/summary.csv line 3 --> |
| results/unified_zeros/gpt2/summary.csv | gpt2 | full | 64301.89 tok/s <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_zeros/gpt2/summary.csv line 2 --> |
| results/unified_zeros/gpt2/summary.csv | gpt2 | zeros_attention | 9186.96 tok/s <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> | 1221.3 MB <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_zeros/gpt2/summary.csv line 3 --> |
| results/unified_zeros/roberta/summary.csv | roberta | full | 80292.04 tok/s <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_zeros/roberta/summary.csv line 2 --> |
| results/unified_zeros/roberta/summary.csv | roberta | zeros_attention | 36002.96 tok/s <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> | 662.04 MB <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_zeros/roberta/summary.csv line 3 --> |
