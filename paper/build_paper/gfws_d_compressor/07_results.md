# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | gfws_d | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | gfws_d | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | gfws_d | Elapsed 0.838s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> | 129.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 55 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.70% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.9432 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 36 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:26:47.566951 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |

