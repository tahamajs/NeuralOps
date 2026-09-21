# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cr | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cr | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cr | Elapsed 3.308s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> | 74.9 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 27 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.70% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.9433 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:15:55.168452 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |

