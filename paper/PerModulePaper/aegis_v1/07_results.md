# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | aegis_v1 | PASS_NO_COMPRESSION | 61.81% <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 0.7209 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | aegis_v1 | PASS_NO_COMPRESSION | 70.99% <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 0.7668 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | aegis_v1 | PASS_NO_COMPRESSION | 79.13% <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 0.8131 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | aegis_v1 | 66955010 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 67741538 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 786528 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | aegis_v1 | 124441344 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 126014400 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 1573056 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | aegis_v1 | 124647170 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 133307330 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 1573056 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | full | 39612.94 tok/s <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v1/distilbert/summary.csv | distilbert | aegis_v1 | 12321.89 tok/s <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 329.48 MB <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v1/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | full | 23751.25 tok/s <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v1/gpt2/summary.csv | gpt2 | aegis_v1 | 9887.94 tok/s <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 589.37 MB <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v1/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | full | 23473.61 tok/s <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 2 --> |
| results/unified_aegis_v1/roberta/summary.csv | roberta | aegis_v1 | 6386.38 tok/s <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 674.79 MB <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v1/roberta/summary.csv line 3 --> |
