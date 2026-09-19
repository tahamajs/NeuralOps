# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_vig/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> |
| results/unified_vig/distilbert/summary.csv | distilbert | vig | PASS | 88.3% <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> | 0.8854 <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> |
| results/unified_vig/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> |
| results/unified_vig/gpt2/summary.csv | gpt2 | vig | PASS | 61.35% <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> | 0.7185 <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> |
| results/unified_vig/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_vig/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_vig/roberta/summary.csv line 2 --> |
| results/unified_vig/roberta/summary.csv | roberta | vig | PASS | 61.24% <!-- source: results/unified_vig/roberta/summary.csv line 3 --> | 0.7188 <!-- source: results/unified_vig/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_vig/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> |
| results/unified_vig/distilbert/summary.csv | distilbert | vig | 66955010 <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> | 67655594 <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> | 1292714 <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> | 0.9896x <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> |
| results/unified_vig/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> |
| results/unified_vig/gpt2/summary.csv | gpt2 | vig | 124441344 <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> | 125842512 <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> | 1402704 <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> | 0.9889x <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> |
| results/unified_vig/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_vig/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_vig/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_vig/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_vig/roberta/summary.csv line 2 --> |
| results/unified_vig/roberta/summary.csv | roberta | vig | 124647170 <!-- source: results/unified_vig/roberta/summary.csv line 3 --> | 133135442 <!-- source: results/unified_vig/roberta/summary.csv line 3 --> | 1993298 <!-- source: results/unified_vig/roberta/summary.csv line 3 --> | 0.9362x <!-- source: results/unified_vig/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_vig/distilbert/summary.csv | distilbert | full | 141848.23 tok/s <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_vig/distilbert/summary.csv line 2 --> |
| results/unified_vig/distilbert/summary.csv | distilbert | vig | 53277.39 tok/s <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> | 348.81 MB <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_vig/distilbert/summary.csv line 3 --> |
| results/unified_vig/gpt2/summary.csv | gpt2 | full | 65606.22 tok/s <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_vig/gpt2/summary.csv line 2 --> |
| results/unified_vig/gpt2/summary.csv | gpt2 | vig | 23507.9 tok/s <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> | 573.93 MB <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_vig/gpt2/summary.csv line 3 --> |
| results/unified_vig/roberta/summary.csv | roberta | full | 74815.97 tok/s <!-- source: results/unified_vig/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_vig/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_vig/roberta/summary.csv line 2 --> |
| results/unified_vig/roberta/summary.csv | roberta | vig | 25687.36 tok/s <!-- source: results/unified_vig/roberta/summary.csv line 3 --> | 679.91 MB <!-- source: results/unified_vig/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_vig/roberta/summary.csv line 3 --> |
