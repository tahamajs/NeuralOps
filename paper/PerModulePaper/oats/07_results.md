# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_oats/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> |
| results/unified_oats/distilbert/summary.csv | distilbert | oats | PASS | 90.02% <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> | 0.9019 <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> |
| results/unified_oats/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> |
| results/unified_oats/gpt2/summary.csv | gpt2 | oats | PASS | 76.03% <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> | 0.7538 <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> |
| results/unified_oats/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_oats/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_oats/roberta/summary.csv line 2 --> |
| results/unified_oats/roberta/summary.csv | roberta | oats | PASS | 69.38% <!-- source: results/unified_oats/roberta/summary.csv line 3 --> | 0.7652 <!-- source: results/unified_oats/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | oats | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_oats/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> |
| results/unified_oats/distilbert/summary.csv | distilbert | oats | 66955010 <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> | 66955046 <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> | 36 <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> |
| results/unified_oats/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> |
| results/unified_oats/gpt2/summary.csv | gpt2 | oats | 124441344 <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> | 124441392 <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> | 48 <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> |
| results/unified_oats/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_oats/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_oats/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_oats/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_oats/roberta/summary.csv line 2 --> |
| results/unified_oats/roberta/summary.csv | roberta | oats | 124647170 <!-- source: results/unified_oats/roberta/summary.csv line 3 --> | 124647242 <!-- source: results/unified_oats/roberta/summary.csv line 3 --> | 72 <!-- source: results/unified_oats/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_oats/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | oats | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_oats/distilbert/summary.csv | distilbert | full | 58966.83 tok/s <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_oats/distilbert/summary.csv line 2 --> |
| results/unified_oats/distilbert/summary.csv | distilbert | oats | 48651.3 tok/s <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> | 226.48 MB <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> | 0.1780456701914469 <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> |
| results/unified_oats/gpt2/summary.csv | gpt2 | full | 26794.54 tok/s <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_oats/gpt2/summary.csv line 2 --> |
| results/unified_oats/gpt2/summary.csv | gpt2 | oats | 24247.27 tok/s <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> | 856.33 MB <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> | 0.1916344678029418 <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> |
| results/unified_oats/roberta/summary.csv | roberta | full | 26961.83 tok/s <!-- source: results/unified_oats/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_oats/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_oats/roberta/summary.csv line 2 --> |
| results/unified_oats/roberta/summary.csv | roberta | oats | 23332.35 tok/s <!-- source: results/unified_oats/roberta/summary.csv line 3 --> | 864.62 MB <!-- source: results/unified_oats/roberta/summary.csv line 3 --> | 0.1721547299789058 <!-- source: results/unified_oats/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | oats | Elapsed 3.735s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> | 294.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.60% | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| F1 Score | 0.9074 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compressed Layers | 36 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Timestamp | 2026-09-19T01:41:40.142021 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [79.36]
> These numbers must NOT be cited in the final paper.

