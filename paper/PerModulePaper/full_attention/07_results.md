# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_full/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_full/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_full/distilbert/summary.csv line 2 --> |
| results/unified_full/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_full/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_full/gpt2/summary.csv line 2 --> |
| results/unified_full/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_full/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_full/roberta/summary.csv line 2 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_full/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_full/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_full/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_full/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_full/distilbert/summary.csv line 2 --> |
| results/unified_full/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_full/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_full/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_full/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_full/gpt2/summary.csv line 2 --> |
| results/unified_full/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_full/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_full/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_full/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_full/roberta/summary.csv line 2 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_full/distilbert/summary.csv | distilbert | full | 53818.36 tok/s <!-- source: results/unified_full/distilbert/summary.csv line 2 --> | 388.01 MB <!-- source: results/unified_full/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_full/distilbert/summary.csv line 2 --> |
| results/unified_full/gpt2/summary.csv | gpt2 | full | 26323.5 tok/s <!-- source: results/unified_full/gpt2/summary.csv line 2 --> | 791.56 MB <!-- source: results/unified_full/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_full/gpt2/summary.csv line 2 --> |
| results/unified_full/roberta/summary.csv | roberta | full | 26612.78 tok/s <!-- source: results/unified_full/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_full/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_full/roberta/summary.csv line 2 --> |
