# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_slicegpt_pca/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> |
| results/unified_slicegpt_pca/distilbert/summary.csv | distilbert | slicegpt_pca | PASS | 90.48% <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> | 0.9077 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> |
| results/unified_slicegpt_pca/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> |
| results/unified_slicegpt_pca/gpt2/summary.csv | gpt2 | slicegpt_pca | PASS | 62.04% <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> | 0.6783 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> |
| results/unified_slicegpt_pca/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> |
| results/unified_slicegpt_pca/roberta/summary.csv | roberta | slicegpt_pca | PASS | 67.66% <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> | 0.7496 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slicegpt_pca | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_slicegpt_pca/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> |
| results/unified_slicegpt_pca/distilbert/summary.csv | distilbert | slicegpt_pca | 66955010 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> |
| results/unified_slicegpt_pca/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> |
| results/unified_slicegpt_pca/gpt2/summary.csv | gpt2 | slicegpt_pca | 124441344 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> |
| results/unified_slicegpt_pca/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> |
| results/unified_slicegpt_pca/roberta/summary.csv | roberta | slicegpt_pca | 124647170 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slicegpt_pca | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_slicegpt_pca/distilbert/summary.csv | distilbert | full | 137657.71 tok/s <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 2 --> |
| results/unified_slicegpt_pca/distilbert/summary.csv | distilbert | slicegpt_pca | 136922.23 tok/s <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> | 312.19 MB <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> | 0.9777368952830632 <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> |
| results/unified_slicegpt_pca/gpt2/summary.csv | gpt2 | full | 63666.99 tok/s <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 2 --> |
| results/unified_slicegpt_pca/gpt2/summary.csv | gpt2 | slicegpt_pca | 67199.32 tok/s <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> | 637.29 MB <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> | 0.9872238064805666 <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> |
| results/unified_slicegpt_pca/roberta/summary.csv | roberta | full | 88423.74 tok/s <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 2 --> |
| results/unified_slicegpt_pca/roberta/summary.csv | roberta | slicegpt_pca | 87752.88 tok/s <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> | 300.97 MB <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> | 0.5473480307393603 <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slicegpt_pca | Elapsed 58.804s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> | 135.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> |
