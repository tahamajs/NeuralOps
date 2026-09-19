# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | smart_weighted_ensemble | PASS | 90.14% <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 0.9029 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | smart_weighted_ensemble | PASS | 61.01% <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 0.6654 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | smart_weighted_ensemble | FAIL | 0.0% <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 0.7586 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | smart_weighted_ensemble | 66955010 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | smart_weighted_ensemble | 124441344 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | smart_weighted_ensemble | 124647170 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | full | 148495.02 tok/s <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/distilbert/summary.csv | distilbert | smart_weighted_ensemble | 144610.54 tok/s <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/distilbert/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | full | 62844.73 tok/s <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/gpt2/summary.csv | gpt2 | smart_weighted_ensemble | 66142.57 tok/s <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/gpt2/summary.csv line 3 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | full | 90470.13 tok/s <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 2 --> |
| results/unified_smart_weighted_ensemble/roberta/summary.csv | roberta | smart_weighted_ensemble | 90470.13 tok/s <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 298.71 MB <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_smart_weighted_ensemble/roberta/summary.csv line 3 --> |
