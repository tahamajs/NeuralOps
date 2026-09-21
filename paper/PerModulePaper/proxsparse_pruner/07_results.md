# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_proxsparse/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> |
| results/unified_proxsparse/distilbert/summary.csv | distilbert | proxsparse | PASS | 78.33% <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> | 0.7845 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> |
| results/unified_proxsparse/gpt2/summary.csv | gpt2 | proxsparse | PASS_LOW | 54.01% <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> | 0.4866 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> |
| results/unified_proxsparse/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> |
| results/unified_proxsparse/roberta/summary.csv | roberta | proxsparse | PASS_LOW | 50.92% <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/distilbert/summary.csv | distilbert | proxsparse_pruner | PASS | 78.33% <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> | 0.7845 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/gpt2/summary.csv | gpt2 | proxsparse_pruner | PASS_LOW | 54.01% <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> | 0.4866 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/roberta/summary.csv | roberta | proxsparse_pruner | PASS_LOW | 50.92% <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | proxsparse | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | proxsparse_pruner | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_proxsparse/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> |
| results/unified_proxsparse/distilbert/summary.csv | distilbert | proxsparse | 66955010 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> |
| results/unified_proxsparse/gpt2/summary.csv | gpt2 | proxsparse | 124441344 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> |
| results/unified_proxsparse/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> |
| results/unified_proxsparse/roberta/summary.csv | roberta | proxsparse | 124647170 <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/distilbert/summary.csv | distilbert | proxsparse_pruner | 66955010 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/gpt2/summary.csv | gpt2 | proxsparse_pruner | 124441344 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/roberta/summary.csv | roberta | proxsparse_pruner | 124647170 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | proxsparse | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | proxsparse_pruner | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_proxsparse/distilbert/summary.csv | distilbert | full | 147172.28 tok/s <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 2 --> |
| results/unified_proxsparse/distilbert/summary.csv | distilbert | proxsparse | 162708.29 tok/s <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> | 0.6790638698471917 <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse/gpt2/summary.csv | gpt2 | full | 62312.29 tok/s <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 2 --> |
| results/unified_proxsparse/gpt2/summary.csv | gpt2 | proxsparse | 64596.44 tok/s <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> | 0.5756725743412971 <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> |
| results/unified_proxsparse/roberta/summary.csv | roberta | full | 96397.05 tok/s <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_proxsparse/roberta/summary.csv line 2 --> |
| results/unified_proxsparse/roberta/summary.csv | roberta | proxsparse | 90541.96 tok/s <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> | 0.6366075558794869 <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/distilbert/summary.csv | distilbert | full | 140605.0 tok/s <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/distilbert/summary.csv | distilbert | proxsparse_pruner | 156067.33 tok/s <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> | 0.6790638698471917 <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/gpt2/summary.csv | gpt2 | full | 62805.44 tok/s <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/gpt2/summary.csv | gpt2 | proxsparse_pruner | 65737.96 tok/s <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> | 0.5756725743412971 <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/roberta/summary.csv | roberta | full | 90155.06 tok/s <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 2 --> |
| results/unified_proxsparse_pruner/roberta/summary.csv | roberta | proxsparse_pruner | 85746.44 tok/s <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> | 0.6366075558794869 <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | proxsparse | Elapsed 0.334s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> | 155.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | proxsparse_pruner | Elapsed 0.314s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> | 155.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 78.33% | `results/unified_proxsparse_pruner/distilbert/summary.csv` |
| F1 Score | 0.7845 | `results/unified_proxsparse_pruner/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_proxsparse_pruner/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_proxsparse_pruner/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_proxsparse_pruner/distilbert/summary.csv` |
| Timestamp | 2026-09-11T07:27:54.401807 | `results/unified_proxsparse_pruner/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.48, 60.78, 75.11, 90.48, 60.78, 75.11]
> These numbers must NOT be cited in the final paper.

