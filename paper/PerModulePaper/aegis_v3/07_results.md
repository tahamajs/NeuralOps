# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | aegis_v3 | PASS_NO_COMPRESSION | 78.44% <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 0.8216 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | aegis_v3 | PASS_NO_COMPRESSION | 54.82% <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 0.5397 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | aegis_v3 | PASS_NO_COMPRESSION | 55.16% <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 0.6943 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | aegis_v3 | 66955010 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 67741538 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 786528 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | aegis_v3 | 124441344 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 126014400 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 1573056 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | aegis_v3 | 124647170 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 133307330 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 1573056 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | full | 51110.39 tok/s <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 2 --> |
| results/unified_aegis_v3/distilbert/summary.csv | distilbert | aegis_v3 | 12530.63 tok/s <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 208.28 MB <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v3/distilbert/summary.csv line 3 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | full | 23517.17 tok/s <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 2 --> |
| results/unified_aegis_v3/gpt2/summary.csv | gpt2 | aegis_v3 | 8863.42 tok/s <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 594.88 MB <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v3/gpt2/summary.csv line 3 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | full | 24095.68 tok/s <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 2 --> |
| results/unified_aegis_v3/roberta/summary.csv | roberta | aegis_v3 | 5952.88 tok/s <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 688.09 MB <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_aegis_v3/roberta/summary.csv line 3 --> |
