# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_zerokv_neo/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> |
| results/unified_zerokv_neo/distilbert/summary.csv | distilbert | zerokv_neo | PASS | 75.92% <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> | 0.7925 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> |
| results/unified_zerokv_neo/gpt2/summary.csv | gpt2 | full | PASS | 87.84% <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> | 0.8776 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> |
| results/unified_zerokv_neo/gpt2/summary.csv | gpt2 | zerokv_neo | PASS_LOW | 51.15% <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> | 0.6758 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> |
| results/unified_zerokv_neo/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> |
| results/unified_zerokv_neo/roberta/summary.csv | roberta | zerokv_neo | PASS_LOW | 50.92% <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | zerokv_neo | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_zerokv_neo/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> |
| results/unified_zerokv_neo/distilbert/summary.csv | distilbert | zerokv_neo | 66955010 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> |
| results/unified_zerokv_neo/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> |
| results/unified_zerokv_neo/gpt2/summary.csv | gpt2 | zerokv_neo | 126800640 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> | 126800640 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> | 2359296 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> |
| results/unified_zerokv_neo/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> |
| results/unified_zerokv_neo/roberta/summary.csv | roberta | zerokv_neo | 124647170 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | zerokv_neo | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_zerokv_neo/distilbert/summary.csv | distilbert | full | 51116.73 tok/s <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 2 --> |
| results/unified_zerokv_neo/distilbert/summary.csv | distilbert | zerokv_neo | 52122.59 tok/s <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> | 312.19 MB <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> | 0.7781301700299946 <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> |
| results/unified_zerokv_neo/gpt2/summary.csv | gpt2 | full | 25032.32 tok/s <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 2 --> |
| results/unified_zerokv_neo/gpt2/summary.csv | gpt2 | zerokv_neo | 14040.51 tok/s <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> | 413.5 MB <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> | 0.539870709905243 <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> |
| results/unified_zerokv_neo/roberta/summary.csv | roberta | full | 23372.01 tok/s <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 2 --> |
| results/unified_zerokv_neo/roberta/summary.csv | roberta | zerokv_neo | 23289.24 tok/s <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> | 531.69 MB <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> | 0.7596934036492637 <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | zerokv_neo | Elapsed 0.177s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> |
