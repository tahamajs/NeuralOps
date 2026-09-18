# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_adaptive_ranks/distilbert/summary.csv | distilbert | adaptive_ranks | PASS_NO_COMPRESSION | 90.94% <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_ranks/gpt2/summary.csv | gpt2 | adaptive_ranks | PASS | 76.95% <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> | 0.8016 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_ranks/qwen3_5/summary.csv | qwen3_5 | adaptive_ranks | PASS | 81.88% <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> | 0.798 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_ranks/roberta/summary.csv | roberta | adaptive_ranks | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | adaptive_ranks | PASS | Comp 0.97x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_adaptive_ranks/distilbert/summary.csv | distilbert | adaptive_ranks | 66955010 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> | 1538 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_ranks/gpt2/summary.csv | gpt2 | adaptive_ranks | 124441344 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> | 124177153 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> | 326401 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> | 1.0021x <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_ranks/qwen3_5/summary.csv | qwen3_5 | adaptive_ranks | 763217728 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> | 10824704 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_ranks/roberta/summary.csv | roberta | adaptive_ranks | 124647170 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | adaptive_ranks | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> | Reduced (0.97x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> | 0.97x <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_adaptive_ranks/distilbert/summary.csv | distilbert | adaptive_ranks | 50959.72 tok/s <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> | 182.45 MB <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> | 0.0 <!-- source: results/unified_adaptive_ranks/distilbert/summary.csv line 2 --> |
| results/unified_adaptive_ranks/gpt2/summary.csv | gpt2 | adaptive_ranks | 23688.0 tok/s <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> | 638.89 MB <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> | 0.1982078150637553 <!-- source: results/unified_adaptive_ranks/gpt2/summary.csv line 2 --> |
| results/unified_adaptive_ranks/qwen3_5/summary.csv | qwen3_5 | adaptive_ranks | 4497.54 tok/s <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> | 1877.96 MB <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adaptive_ranks/qwen3_5/summary.csv line 2 --> |
| results/unified_adaptive_ranks/roberta/summary.csv | roberta | adaptive_ranks | 23773.5 tok/s <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> | 520.18 MB <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> | 0.0 <!-- source: results/unified_adaptive_ranks/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | adaptive_ranks | Elapsed 0.932s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> | 168.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 7 --> |
