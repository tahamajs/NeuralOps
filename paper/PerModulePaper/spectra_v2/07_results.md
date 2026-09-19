# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_spectra_v2/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> |
| results/unified_spectra_v2/distilbert/summary.csv | distilbert | spectra_v2 | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> |
| results/unified_spectra_v2/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> |
| results/unified_spectra_v2/gpt2/summary.csv | gpt2 | spectra_v2 | PASS_NO_COMPRESSION | 79.13% <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> | 0.7955 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> |
| results/unified_spectra_v2/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> |
| results/unified_spectra_v2/roberta/summary.csv | roberta | spectra_v2 | PASS_NO_COMPRESSION | 80.28% <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> | 0.8197 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_spectra_v2/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> |
| results/unified_spectra_v2/distilbert/summary.csv | distilbert | spectra_v2 | 66955010 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> | 67544858 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> | 589848 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> |
| results/unified_spectra_v2/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> |
| results/unified_spectra_v2/gpt2/summary.csv | gpt2 | spectra_v2 | 124441344 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> | 125621040 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> | 1179696 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> |
| results/unified_spectra_v2/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> |
| results/unified_spectra_v2/roberta/summary.csv | roberta | spectra_v2 | 124647170 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> | 132913970 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> | 1179696 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_spectra_v2/distilbert/summary.csv | distilbert | full | 28405.95 tok/s <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 2 --> |
| results/unified_spectra_v2/distilbert/summary.csv | distilbert | spectra_v2 | 14003.98 tok/s <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> | 324.95 MB <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectra_v2/distilbert/summary.csv line 3 --> |
| results/unified_spectra_v2/gpt2/summary.csv | gpt2 | full | 17589.97 tok/s <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 2 --> |
| results/unified_spectra_v2/gpt2/summary.csv | gpt2 | spectra_v2 | 6899.61 tok/s <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> | 567.7 MB <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectra_v2/gpt2/summary.csv line 3 --> |
| results/unified_spectra_v2/roberta/summary.csv | roberta | full | 15681.02 tok/s <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 2 --> |
| results/unified_spectra_v2/roberta/summary.csv | roberta | spectra_v2 | 6917.18 tok/s <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> | 655.54 MB <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_spectra_v2/roberta/summary.csv line 3 --> |
