# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_hgrn2_outer/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> |
| results/unified_hgrn2_outer/distilbert/summary.csv | distilbert | hgrn2_outer | PASS | 68.46% <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> | 0.7114 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> |
| results/unified_hgrn2_outer/gpt2/summary.csv | gpt2 | full | PASS | 57.91% <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> | 0.301 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> |
| results/unified_hgrn2_outer/gpt2/summary.csv | gpt2 | hgrn2_outer | PASS_LOW | 48.62% <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> | 0.6203 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> |
| results/unified_hgrn2_outer/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> |
| results/unified_hgrn2_outer/roberta/summary.csv | roberta | hgrn2_outer | PASS_LOW | 50.34% <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> | 0.4355 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_hgrn2_outer/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> |
| results/unified_hgrn2_outer/distilbert/summary.csv | distilbert | hgrn2_outer | 66955010 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> | 67587176 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> | 1181984 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> | 0.9906x <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> |
| results/unified_hgrn2_outer/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> |
| results/unified_hgrn2_outer/gpt2/summary.csv | gpt2 | hgrn2_outer | 124441344 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> | 125705676 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> | 1181244 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> | 0.9899x <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> |
| results/unified_hgrn2_outer/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> |
| results/unified_hgrn2_outer/roberta/summary.csv | roberta | hgrn2_outer | 124647170 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> | 132998606 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> | 1771838 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> | 0.9372x <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_hgrn2_outer/distilbert/summary.csv | distilbert | full | 172375.58 tok/s <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 2 --> |
| results/unified_hgrn2_outer/distilbert/summary.csv | distilbert | hgrn2_outer | 2168.02 tok/s <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> | 329.98 MB <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> |
| results/unified_hgrn2_outer/gpt2/summary.csv | gpt2 | full | 27716.2 tok/s <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 2 --> |
| results/unified_hgrn2_outer/gpt2/summary.csv | gpt2 | hgrn2_outer | 9386.81 tok/s <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> | 569.81 MB <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> |
| results/unified_hgrn2_outer/roberta/summary.csv | roberta | full | 92775.9 tok/s <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 2 --> |
| results/unified_hgrn2_outer/roberta/summary.csv | roberta | hgrn2_outer | 8864.8 tok/s <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> | 685.64 MB <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> |
