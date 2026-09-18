# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_direct_svd/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> |
| results/unified_direct_svd/distilbert/summary.csv | distilbert | direct_svd | PASS | 88.99% <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> | 0.8977 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> |
| results/unified_direct_svd/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> |
| results/unified_direct_svd/gpt2/summary.csv | gpt2 | direct_svd | PASS | 68.46% <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> | 0.5827 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> |
| results/unified_direct_svd/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> |
| results/unified_direct_svd/roberta/summary.csv | roberta | direct_svd | PASS | 71.1% <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> | 0.7544 <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | direct_svd | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_direct_svd/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> |
| results/unified_direct_svd/distilbert/summary.csv | distilbert | direct_svd | 66955010 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> | 65333005 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> | 4874507 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> | 1.0248x <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> |
| results/unified_direct_svd/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> |
| results/unified_direct_svd/gpt2/summary.csv | gpt2 | direct_svd | 124441344 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> | 123409158 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> | 4614 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> | 1.0084x <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> |
| results/unified_direct_svd/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> |
| results/unified_direct_svd/roberta/summary.csv | roberta | direct_svd | 124647170 <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> | 121255705 <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> | 17687 <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> | 1.028x <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | direct_svd | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_direct_svd/distilbert/summary.csv | distilbert | full | 51084.83 tok/s <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 2 --> |
| results/unified_direct_svd/distilbert/summary.csv | distilbert | direct_svd | 43766.17 tok/s <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> | 355.29 MB <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> | 0.3280525853991574 <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> |
| results/unified_direct_svd/gpt2/summary.csv | gpt2 | full | 26772.56 tok/s <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 2 --> |
| results/unified_direct_svd/gpt2/summary.csv | gpt2 | direct_svd | 25639.93 tok/s <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> | 646.14 MB <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> | 0.3114239479225169 <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> |
| results/unified_direct_svd/roberta/summary.csv | roberta | full | 27189.71 tok/s <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_direct_svd/roberta/summary.csv line 2 --> |
| results/unified_direct_svd/roberta/summary.csv | roberta | direct_svd | 22488.27 tok/s <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> | 567.52 MB <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> | 0.333649210018602 <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | direct_svd | Elapsed 0.776s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> |
