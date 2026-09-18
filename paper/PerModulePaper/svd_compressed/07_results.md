# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_svd_compressed/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> |
| results/unified_svd_compressed/distilbert/summary.csv | distilbert | svd_compressed | PASS | 87.39% <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> | 0.8825 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> |
| results/unified_svd_compressed/gpt2/summary.csv | gpt2 | full | PASS | 87.84% <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> | 0.8776 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> |
| results/unified_svd_compressed/gpt2/summary.csv | gpt2 | svd_compressed | PASS | 76.15% <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> | 0.7965 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> |
| results/unified_svd_compressed/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> |
| results/unified_svd_compressed/roberta/summary.csv | roberta | svd_compressed | PASS | 58.14% <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> | 0.703 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_svd_compressed/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> |
| results/unified_svd_compressed/distilbert/summary.csv | distilbert | svd_compressed | 66955010 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> | 58107650 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> | 589824 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> | 1.1523x <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> |
| results/unified_svd_compressed/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> |
| results/unified_svd_compressed/gpt2/summary.csv | gpt2 | svd_compressed | 126800640 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> | 108221196 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> | 1474572 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> | 1.1717x <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> |
| results/unified_svd_compressed/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> |
| results/unified_svd_compressed/roberta/summary.csv | roberta | svd_compressed | 124647170 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> | 106961666 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> | 1179648 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> | 1.1653x <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_svd_compressed/distilbert/summary.csv | distilbert | full | 12606.61 tok/s <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> | 388.55 MB <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 --> |
| results/unified_svd_compressed/distilbert/summary.csv | distilbert | svd_compressed | 5973.8 tok/s <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> | 283.97 MB <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 3 --> |
| results/unified_svd_compressed/gpt2/summary.csv | gpt2 | full | 27219.81 tok/s <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 2 --> |
| results/unified_svd_compressed/gpt2/summary.csv | gpt2 | svd_compressed | 10244.56 tok/s <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> | 300.29 MB <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_compressed/gpt2/summary.csv line 3 --> |
| results/unified_svd_compressed/roberta/summary.csv | roberta | full | 25619.66 tok/s <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 2 --> |
| results/unified_svd_compressed/roberta/summary.csv | roberta | svd_compressed | 14075.4 tok/s <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> | 553.0 MB <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_svd_compressed/roberta/summary.csv line 3 --> |
