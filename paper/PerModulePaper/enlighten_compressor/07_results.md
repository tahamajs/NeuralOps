# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_enlighten/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> |
| results/unified_enlighten/distilbert/summary.csv | distilbert | enlighten | PASS | 89.22% <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> | 0.8937 <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> |
| results/unified_enlighten/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> |
| results/unified_enlighten/gpt2/summary.csv | gpt2 | enlighten | PASS | 61.12% <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> | 0.3936 <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> |
| results/unified_enlighten/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> |
| results/unified_enlighten/roberta/summary.csv | roberta | enlighten | PASS_LOW | 51.03% <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | enlighten | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_enlighten/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> |
| results/unified_enlighten/distilbert/summary.csv | distilbert | enlighten | 66955010 <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> |
| results/unified_enlighten/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> |
| results/unified_enlighten/gpt2/summary.csv | gpt2 | enlighten | 124441344 <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> |
| results/unified_enlighten/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> |
| results/unified_enlighten/roberta/summary.csv | roberta | enlighten | 124647170 <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | enlighten | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_enlighten/distilbert/summary.csv | distilbert | full | 52725.06 tok/s <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_enlighten/distilbert/summary.csv line 2 --> |
| results/unified_enlighten/distilbert/summary.csv | distilbert | enlighten | 53181.36 tok/s <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> | 301.18 MB <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> | 0.2407765909407657 <!-- source: results/unified_enlighten/distilbert/summary.csv line 3 --> |
| results/unified_enlighten/gpt2/summary.csv | gpt2 | full | 26923.15 tok/s <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_enlighten/gpt2/summary.csv line 2 --> |
| results/unified_enlighten/gpt2/summary.csv | gpt2 | enlighten | 25776.19 tok/s <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> | 628.28 MB <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> | 0.2547934372152875 <!-- source: results/unified_enlighten/gpt2/summary.csv line 3 --> |
| results/unified_enlighten/roberta/summary.csv | roberta | full | 27357.52 tok/s <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_enlighten/roberta/summary.csv line 2 --> |
| results/unified_enlighten/roberta/summary.csv | roberta | enlighten | 26019.95 tok/s <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> | 520.68 MB <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> | 0.2388048979619952 <!-- source: results/unified_enlighten/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | enlighten | Elapsed 2.36s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> | 152.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 40 --> |
