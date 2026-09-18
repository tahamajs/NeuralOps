# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_nmf/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> |
| results/unified_nmf/distilbert/summary.csv | distilbert | nmf | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> |
| results/unified_nmf/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> |
| results/unified_nmf/gpt2/summary.csv | gpt2 | nmf | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> |
| results/unified_nmf/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> |
| results/unified_nmf/roberta/summary.csv | roberta | nmf | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nmf | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nmf_compressor | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_nmf/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> |
| results/unified_nmf/distilbert/summary.csv | distilbert | nmf | 66955010 <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> |
| results/unified_nmf/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> |
| results/unified_nmf/gpt2/summary.csv | gpt2 | nmf | 124441344 <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> |
| results/unified_nmf/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> |
| results/unified_nmf/roberta/summary.csv | roberta | nmf | 124647170 <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nmf | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nmf_compressor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_nmf/distilbert/summary.csv | distilbert | full | 52348.76 tok/s <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_nmf/distilbert/summary.csv line 2 --> |
| results/unified_nmf/distilbert/summary.csv | distilbert | nmf | 48602.9 tok/s <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> |
| results/unified_nmf/gpt2/summary.csv | gpt2 | full | 19888.29 tok/s <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_nmf/gpt2/summary.csv line 2 --> |
| results/unified_nmf/gpt2/summary.csv | gpt2 | nmf | 24984.51 tok/s <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> | 626.91 MB <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> |
| results/unified_nmf/roberta/summary.csv | roberta | full | 27371.76 tok/s <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_nmf/roberta/summary.csv line 2 --> |
| results/unified_nmf/roberta/summary.csv | roberta | nmf | 26550.6 tok/s <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nmf | Elapsed 9.873s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> | 101.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nmf_compressor | Elapsed 9.967s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> | 101.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> |
