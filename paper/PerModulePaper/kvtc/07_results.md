# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_kvtc/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> |
| results/unified_kvtc/distilbert/summary.csv | distilbert | kvtc | PASS | 90.14% <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> | 0.9025 <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> |
| results/unified_kvtc/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> |
| results/unified_kvtc/gpt2/summary.csv | gpt2 | kvtc | FAIL | 0.0% <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> |
| results/unified_kvtc/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> |
| results/unified_kvtc/roberta/summary.csv | roberta | kvtc | PASS_LOW | 50.92% <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kvtc | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_kvtc/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> |
| results/unified_kvtc/distilbert/summary.csv | distilbert | kvtc | 66955010 <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> | 66955030 <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> | 20 <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> |
| results/unified_kvtc/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> |
| results/unified_kvtc/gpt2/summary.csv | gpt2 | kvtc | 124441344 <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> |
| results/unified_kvtc/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> |
| results/unified_kvtc/roberta/summary.csv | roberta | kvtc | 124647170 <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> | 124647215 <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> | 7679279 <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kvtc | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_kvtc/distilbert/summary.csv | distilbert | full | 59024.72 tok/s <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kvtc/distilbert/summary.csv line 2 --> |
| results/unified_kvtc/distilbert/summary.csv | distilbert | kvtc | 51750.86 tok/s <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> | 217.23 MB <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> | 0.366407522559166 <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> |
| results/unified_kvtc/gpt2/summary.csv | gpt2 | full | 26966.76 tok/s <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kvtc/gpt2/summary.csv line 2 --> |
| results/unified_kvtc/gpt2/summary.csv | gpt2 | kvtc | 26966.76 tok/s <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> |
| results/unified_kvtc/roberta/summary.csv | roberta | full | 27404.83 tok/s <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_kvtc/roberta/summary.csv line 2 --> |
| results/unified_kvtc/roberta/summary.csv | roberta | kvtc | 97731.31 tok/s <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> | 457.87 MB <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> | 0.3985868076483408 <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kvtc | Elapsed 0.789s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> | 147.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> |
