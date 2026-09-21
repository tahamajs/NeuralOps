# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_subspace_projection/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> |
| results/unified_subspace_projection/distilbert/summary.csv | distilbert | subspace_projection | PASS | 89.91% <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> | 0.8993 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> |
| results/unified_subspace_projection/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> |
| results/unified_subspace_projection/gpt2/summary.csv | gpt2 | subspace_projection | PASS | 59.52% <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> | 0.6596 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> |
| results/unified_subspace_projection/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> |
| results/unified_subspace_projection/roberta/summary.csv | roberta | subspace_projection | PASS | 84.17% <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> | 0.8535 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | subspace_projection | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_subspace_projection/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> |
| results/unified_subspace_projection/distilbert/summary.csv | distilbert | subspace_projection | 66955010 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> |
| results/unified_subspace_projection/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> |
| results/unified_subspace_projection/gpt2/summary.csv | gpt2 | subspace_projection | 124441344 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> | 124048129 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> | 198145 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> | 1.0032x <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> |
| results/unified_subspace_projection/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> |
| results/unified_subspace_projection/roberta/summary.csv | roberta | subspace_projection | 124647170 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | subspace_projection | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_subspace_projection/distilbert/summary.csv | distilbert | full | 144469.77 tok/s <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 2 --> |
| results/unified_subspace_projection/distilbert/summary.csv | distilbert | subspace_projection | 140656.98 tok/s <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> | 338.84 MB <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> |
| results/unified_subspace_projection/gpt2/summary.csv | gpt2 | full | 63965.92 tok/s <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 2 --> |
| results/unified_subspace_projection/gpt2/summary.csv | gpt2 | subspace_projection | 64096.61 tok/s <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> | 637.29 MB <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> |
| results/unified_subspace_projection/roberta/summary.csv | roberta | full | 99998.19 tok/s <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 2 --> |
| results/unified_subspace_projection/roberta/summary.csv | roberta | subspace_projection | 83248.79 tok/s <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> | 364.29 MB <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | subspace_projection | Elapsed 0.757s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> | 115.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=59.52%, comp=1.003x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 59.52% | `results/unified_subspace_projection/gpt2/summary.csv` |
| F1 Score | 0.6596 | `results/unified_subspace_projection/gpt2/summary.csv` |
| Compression | 1.003x | `results/unified_subspace_projection/gpt2/summary.csv` |
| Compressed Layers | 0 | `results/unified_subspace_projection/gpt2/summary.csv` |
| Fallback Layers | 0 | `results/unified_subspace_projection/gpt2/summary.csv` |
| Timestamp | 2026-09-11T08:48:45.869618 | `results/unified_subspace_projection/gpt2/summary.csv` |

