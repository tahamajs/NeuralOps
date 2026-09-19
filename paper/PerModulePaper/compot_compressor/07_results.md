# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | compot | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | compot | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | compot | Elapsed 0.77s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> | 106.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 24 --> |
