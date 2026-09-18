# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_qcqa/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> |
| results/unified_qcqa/distilbert/summary.csv | distilbert | qcqa | PASS | 63.76% <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> | 0.7299 <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> |
| results/unified_qcqa/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> |
| results/unified_qcqa/gpt2/summary.csv | gpt2 | qcqa | PASS_LOW | 50.92% <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> | 0.6738 <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> |
| results/unified_qcqa/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> |
| results/unified_qcqa/roberta/summary.csv | roberta | qcqa | PASS_LOW | 51.03% <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_qcqa/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> |
| results/unified_qcqa/distilbert/summary.csv | distilbert | qcqa | 66955010 <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> | 67549466 <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> | 1181978 <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> | 0.9912x <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> |
| results/unified_qcqa/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> |
| results/unified_qcqa/gpt2/summary.csv | gpt2 | qcqa | 124441344 <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> | 125630256 <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> | 1181232 <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> | 0.9905x <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> |
| results/unified_qcqa/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> |
| results/unified_qcqa/roberta/summary.csv | roberta | qcqa | 124647170 <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> | 132923186 <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> | 1771826 <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> | 0.9377x <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_qcqa/distilbert/summary.csv | distilbert | full | 146754.83 tok/s <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_qcqa/distilbert/summary.csv line 2 --> |
| results/unified_qcqa/distilbert/summary.csv | distilbert | qcqa | 67004.99 tok/s <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> | 334.32 MB <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_qcqa/distilbert/summary.csv line 3 --> |
| results/unified_qcqa/gpt2/summary.csv | gpt2 | full | 62665.12 tok/s <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_qcqa/gpt2/summary.csv line 2 --> |
| results/unified_qcqa/gpt2/summary.csv | gpt2 | qcqa | 31268.37 tok/s <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> | 568.75 MB <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_qcqa/gpt2/summary.csv line 3 --> |
| results/unified_qcqa/roberta/summary.csv | roberta | full | 96386.47 tok/s <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_qcqa/roberta/summary.csv line 2 --> |
| results/unified_qcqa/roberta/summary.csv | roberta | qcqa | 34912.42 tok/s <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> | 382.56 MB <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_qcqa/roberta/summary.csv line 3 --> |
