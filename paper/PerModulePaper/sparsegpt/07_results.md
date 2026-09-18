# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_sparsegpt/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> |
| results/unified_sparsegpt/distilbert/summary.csv | distilbert | sparsegpt | PASS | 86.93% <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> | 0.8747 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> |
| results/unified_sparsegpt/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> |
| results/unified_sparsegpt/gpt2/summary.csv | gpt2 | sparsegpt | PASS_LOW | 53.33% <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> | 0.6661 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> |
| results/unified_sparsegpt/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> |
| results/unified_sparsegpt/roberta/summary.csv | roberta | sparsegpt | PASS_LOW | 50.92% <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | sparsegpt | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_sparsegpt/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> |
| results/unified_sparsegpt/distilbert/summary.csv | distilbert | sparsegpt | 66955010 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> |
| results/unified_sparsegpt/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> |
| results/unified_sparsegpt/gpt2/summary.csv | gpt2 | sparsegpt | 124441344 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> |
| results/unified_sparsegpt/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> |
| results/unified_sparsegpt/roberta/summary.csv | roberta | sparsegpt | 124647170 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | sparsegpt | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_sparsegpt/distilbert/summary.csv | distilbert | full | 153565.86 tok/s <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 2 --> |
| results/unified_sparsegpt/distilbert/summary.csv | distilbert | sparsegpt | 152277.21 tok/s <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> | 340.09 MB <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> |
| results/unified_sparsegpt/gpt2/summary.csv | gpt2 | full | 62506.97 tok/s <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 2 --> |
| results/unified_sparsegpt/gpt2/summary.csv | gpt2 | sparsegpt | 65894.01 tok/s <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> | 634.41 MB <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> |
| results/unified_sparsegpt/roberta/summary.csv | roberta | full | 92299.6 tok/s <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 2 --> |
| results/unified_sparsegpt/roberta/summary.csv | roberta | sparsegpt | 85441.76 tok/s <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> | 363.54 MB <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | sparsegpt | Elapsed 7.585s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> | 222.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> |
