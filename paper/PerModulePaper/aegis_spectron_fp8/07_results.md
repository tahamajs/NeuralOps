# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_aegis_spectron_fp8/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/distilbert/summary.csv | distilbert | aegis_spectron_fp8 | PASS | 79.59% <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 0.8118 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> |
| results/unified_aegis_spectron_fp8/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/gpt2/summary.csv | gpt2 | aegis_spectron_fp8 | PASS | 58.83% <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 0.6913 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> |
| results/unified_aegis_spectron_fp8/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/roberta/summary.csv | roberta | aegis_spectron_fp8 | PASS_LOW | 49.43% <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 0.3815 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_aegis_spectron_fp8/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/distilbert/summary.csv | distilbert | aegis_spectron_fp8 | 66955010 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 54753068 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 760344 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 1.2229x <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> |
| results/unified_aegis_spectron_fp8/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/gpt2/summary.csv | gpt2 | aegis_spectron_fp8 | 124441344 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 100037460 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 1520688 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 1.2439x <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> |
| results/unified_aegis_spectron_fp8/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/roberta/summary.csv | roberta | aegis_spectron_fp8 | 124647170 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 107330390 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 1520688 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 1.1613x <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_aegis_spectron_fp8/distilbert/summary.csv | distilbert | full | 39628.79 tok/s <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/distilbert/summary.csv | distilbert | aegis_spectron_fp8 | 9858.72 tok/s <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 290.29 MB <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_spectron_fp8/distilbert/summary.csv line 3 --> |
| results/unified_aegis_spectron_fp8/gpt2/summary.csv | gpt2 | full | 23426.91 tok/s <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/gpt2/summary.csv | gpt2 | aegis_spectron_fp8 | 5217.46 tok/s <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 643.49 MB <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_spectron_fp8/gpt2/summary.csv line 3 --> |
| results/unified_aegis_spectron_fp8/roberta/summary.csv | roberta | full | 23592.51 tok/s <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 2 --> |
| results/unified_aegis_spectron_fp8/roberta/summary.csv | roberta | aegis_spectron_fp8 | 4840.56 tok/s <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 572.68 MB <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_spectron_fp8/roberta/summary.csv line 3 --> |
