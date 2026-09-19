# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_dobi/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> |
| results/unified_dobi/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> |
| results/unified_dobi/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> |
| results/unified_dobi/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dobi | PASS | Comp 30.95x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_dobi/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> |
| results/unified_dobi/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> |
| results/unified_dobi/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> |
| results/unified_dobi/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dobi | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> | Reduced (30.95x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> | 30.95x <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_dobi/distilbert/summary.csv | distilbert | full | 50749.29 tok/s <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_dobi/distilbert/summary.csv line 2 --> |
| results/unified_dobi/gpt2/summary.csv | gpt2 | full | 26758.23 tok/s <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_dobi/gpt2/summary.csv line 2 --> |
| results/unified_dobi/qwen3_5/summary.csv | qwen3_5 | full | 5935.34 tok/s <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_dobi/qwen3_5/summary.csv line 2 --> |
| results/unified_dobi/roberta/summary.csv | roberta | full | 25886.74 tok/s <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_dobi/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dobi | Elapsed 0.794s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> |
