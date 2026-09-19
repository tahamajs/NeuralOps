# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_lillama/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> |
| results/unified_lillama/distilbert/summary.csv | distilbert | lillama | PASS | 87.5% <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> | 0.8868 <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> |
| results/unified_lillama/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> |
| results/unified_lillama/gpt2/summary.csv | gpt2 | lillama | PASS | 56.88% <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> | 0.6835 <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> |
| results/unified_lillama/roberta/summary.csv | roberta | full | PASS | 75.0% <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> | 0.7578 <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> |
| results/unified_lillama/roberta/summary.csv | roberta | lillama | PASS | 85.44% <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> | 0.8603 <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lillama | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lillama_compressor | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_lillama/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> |
| results/unified_lillama/distilbert/summary.csv | distilbert | lillama | 66955010 <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> |
| results/unified_lillama/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> |
| results/unified_lillama/gpt2/summary.csv | gpt2 | lillama | 124441344 <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> |
| results/unified_lillama/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> |
| results/unified_lillama/roberta/summary.csv | roberta | lillama | 124647170 <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lillama | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lillama_compressor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_lillama/distilbert/summary.csv | distilbert | full | 178757.92 tok/s <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lillama/distilbert/summary.csv line 2 --> |
| results/unified_lillama/distilbert/summary.csv | distilbert | lillama | 187624.63 tok/s <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> |
| results/unified_lillama/gpt2/summary.csv | gpt2 | full | 70978.21 tok/s <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lillama/gpt2/summary.csv line 2 --> |
| results/unified_lillama/gpt2/summary.csv | gpt2 | lillama | 70009.59 tok/s <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> |
| results/unified_lillama/roberta/summary.csv | roberta | full | 97968.75 tok/s <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lillama/roberta/summary.csv line 2 --> |
| results/unified_lillama/roberta/summary.csv | roberta | lillama | 100764.43 tok/s <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> | 583.49 MB <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lillama | Elapsed 0.406s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> | 129.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lillama_compressor | Elapsed 0.426s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> | 129.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> |
