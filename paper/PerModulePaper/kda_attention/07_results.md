# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_kda/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> |
| results/unified_kda/distilbert/summary.csv | distilbert | kda | PASS | 76.83% <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> | 0.7531 <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> |
| results/unified_kda/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> |
| results/unified_kda/gpt2/summary.csv | gpt2 | kda | FAIL | 0.0% <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> |
| results/unified_kda/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_kda/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_kda/roberta/summary.csv line 2 --> |
| results/unified_kda/roberta/summary.csv | roberta | kda | FAIL | 0.0% <!-- source: results/unified_kda/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_kda/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_kda/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> |
| results/unified_kda/distilbert/summary.csv | distilbert | kda | 66955010 <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> | 67544912 <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> | 589902 <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> |
| results/unified_kda/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> |
| results/unified_kda/gpt2/summary.csv | gpt2 | kda | 124441344 <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> |
| results/unified_kda/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_kda/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_kda/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_kda/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kda/roberta/summary.csv line 2 --> |
| results/unified_kda/roberta/summary.csv | roberta | kda | 124647170 <!-- source: results/unified_kda/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_kda/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_kda/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_kda/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_kda/distilbert/summary.csv | distilbert | full | 59839.22 tok/s <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kda/distilbert/summary.csv line 2 --> |
| results/unified_kda/distilbert/summary.csv | distilbert | kda | 18396.52 tok/s <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> | 244.17 MB <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_kda/distilbert/summary.csv line 3 --> |
| results/unified_kda/gpt2/summary.csv | gpt2 | full | 27061.74 tok/s <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kda/gpt2/summary.csv line 2 --> |
| results/unified_kda/gpt2/summary.csv | gpt2 | kda | 27061.74 tok/s <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_kda/gpt2/summary.csv line 3 --> |
| results/unified_kda/roberta/summary.csv | roberta | full | 27262.33 tok/s <!-- source: results/unified_kda/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_kda/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kda/roberta/summary.csv line 2 --> |
| results/unified_kda/roberta/summary.csv | roberta | kda | 27262.33 tok/s <!-- source: results/unified_kda/roberta/summary.csv line 3 --> | 712.34 MB <!-- source: results/unified_kda/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_kda/roberta/summary.csv line 3 --> |
