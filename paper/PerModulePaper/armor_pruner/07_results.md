# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_armor/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> |
| results/unified_armor/distilbert/summary.csv | distilbert | armor_pruner | PASS_NO_COMPRESSION | 90.02% <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> | 0.9032 <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> |
| results/unified_armor/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> |
| results/unified_armor/gpt2/summary.csv | gpt2 | armor_pruner | PASS_NO_COMPRESSION | 49.31% <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> | 0.009 <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> |
| results/unified_armor/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.55% <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> | 0.5501 <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> |
| results/unified_armor/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_armor/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_armor/roberta/summary.csv line 2 --> |
| results/unified_armor/roberta/summary.csv | roberta | armor_pruner | PASS_NO_COMPRESSION | 79.7% <!-- source: results/unified_armor/roberta/summary.csv line 3 --> | 0.8127 <!-- source: results/unified_armor/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | armor | PASS | Comp 0.98x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | armor_pruner | PASS | Comp 0.98x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_armor/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> |
| results/unified_armor/distilbert/summary.csv | distilbert | armor_pruner | 66955010 <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> |
| results/unified_armor/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> |
| results/unified_armor/gpt2/summary.csv | gpt2 | armor_pruner | 124441344 <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> |
| results/unified_armor/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> |
| results/unified_armor/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_armor/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_armor/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_armor/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_armor/roberta/summary.csv line 2 --> |
| results/unified_armor/roberta/summary.csv | roberta | armor_pruner | 124647170 <!-- source: results/unified_armor/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_armor/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_armor/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_armor/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | armor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> | Reduced (0.98x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> | 0.98x <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | armor_pruner | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> | Reduced (0.98x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> | 0.98x <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_armor/distilbert/summary.csv | distilbert | full | 50954.53 tok/s <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_armor/distilbert/summary.csv line 2 --> |
| results/unified_armor/distilbert/summary.csv | distilbert | armor_pruner | 38268.35 tok/s <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> | 0.3922593709495332 <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> |
| results/unified_armor/gpt2/summary.csv | gpt2 | full | 23492.58 tok/s <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_armor/gpt2/summary.csv line 2 --> |
| results/unified_armor/gpt2/summary.csv | gpt2 | armor_pruner | 24010.9 tok/s <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> | 627.91 MB <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> | 0.3875877490888039 <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> |
| results/unified_armor/qwen3_5/summary.csv | qwen3_5 | full | 1279.57 tok/s <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> | 1810.59 MB <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_armor/qwen3_5/summary.csv line 2 --> |
| results/unified_armor/roberta/summary.csv | roberta | full | 21493.88 tok/s <!-- source: results/unified_armor/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_armor/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_armor/roberta/summary.csv line 2 --> |
| results/unified_armor/roberta/summary.csv | roberta | armor_pruner | 23855.46 tok/s <!-- source: results/unified_armor/roberta/summary.csv line 3 --> | 520.68 MB <!-- source: results/unified_armor/roberta/summary.csv line 3 --> | 0.4025514241721895 <!-- source: results/unified_armor/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | armor | Elapsed 0.835s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> | 279.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 13 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | armor_pruner | Elapsed 0.838s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> | 279.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 14 --> |
