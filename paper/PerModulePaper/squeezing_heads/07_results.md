# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_squeezing_heads/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> |
| results/unified_squeezing_heads/distilbert/summary.csv | distilbert | squeezing_heads | PASS | 90.6% <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> | 0.9085 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> |
| results/unified_squeezing_heads/gpt2/summary.csv | gpt2 | full | PASS_LOW | 49.54% <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> | 0.2903 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> |
| results/unified_squeezing_heads/gpt2/summary.csv | gpt2 | squeezing_heads | PASS_LOW | 53.67% <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> | 0.4245 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> |
| results/unified_squeezing_heads/roberta/summary.csv | roberta | full | PASS_LOW | 50.92% <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> | 0.6748 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> |
| results/unified_squeezing_heads/roberta/summary.csv | roberta | squeezing_heads | PASS_LOW | 50.92% <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_squeezing_heads/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> |
| results/unified_squeezing_heads/distilbert/summary.csv | distilbert | squeezing_heads | 66955010 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> | 67544834 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> | 589824 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> | 0.9913x <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> |
| results/unified_squeezing_heads/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> |
| results/unified_squeezing_heads/gpt2/summary.csv | gpt2 | squeezing_heads | 124441344 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> | 125620992 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> | 1179648 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> | 0.9906x <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> |
| results/unified_squeezing_heads/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> |
| results/unified_squeezing_heads/roberta/summary.csv | roberta | squeezing_heads | 124647170 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> | 132913922 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> | 1179648 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> | 0.9378x <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_squeezing_heads/distilbert/summary.csv | distilbert | full | 50688.75 tok/s <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 2 --> |
| results/unified_squeezing_heads/distilbert/summary.csv | distilbert | squeezing_heads | 27497.71 tok/s <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> | 196.03 MB <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_squeezing_heads/distilbert/summary.csv line 3 --> |
| results/unified_squeezing_heads/gpt2/summary.csv | gpt2 | full | 27466.16 tok/s <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> | 390.24 MB <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 2 --> |
| results/unified_squeezing_heads/gpt2/summary.csv | gpt2 | squeezing_heads | 12147.68 tok/s <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> | 325.93 MB <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_squeezing_heads/gpt2/summary.csv line 3 --> |
| results/unified_squeezing_heads/roberta/summary.csv | roberta | full | 27118.79 tok/s <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 2 --> |
| results/unified_squeezing_heads/roberta/summary.csv | roberta | squeezing_heads | 14971.31 tok/s <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> | 371.96 MB <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_squeezing_heads/roberta/summary.csv line 3 --> |
