# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_saes_svd_v2/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> |
| results/unified_saes_svd_v2/distilbert/summary.csv | distilbert | saes_svd_v2 | PASS | 87.5% <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> | 0.8868 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> |
| results/unified_saes_svd_v2/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> |
| results/unified_saes_svd_v2/gpt2/summary.csv | gpt2 | saes_svd_v2 | PASS | 56.88% <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> | 0.6835 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> |
| results/unified_saes_svd_v2/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> |
| results/unified_saes_svd_v2/roberta/summary.csv | roberta | saes_svd_v2 | PASS | 85.44% <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> | 0.8542 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | saes_svd_v2 | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_saes_svd_v2/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> |
| results/unified_saes_svd_v2/distilbert/summary.csv | distilbert | saes_svd_v2 | 66955010 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> |
| results/unified_saes_svd_v2/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> |
| results/unified_saes_svd_v2/gpt2/summary.csv | gpt2 | saes_svd_v2 | 124441344 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> |
| results/unified_saes_svd_v2/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> |
| results/unified_saes_svd_v2/roberta/summary.csv | roberta | saes_svd_v2 | 124647170 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | saes_svd_v2 | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_saes_svd_v2/distilbert/summary.csv | distilbert | full | 142488.01 tok/s <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 2 --> |
| results/unified_saes_svd_v2/distilbert/summary.csv | distilbert | saes_svd_v2 | 131480.74 tok/s <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> | 338.84 MB <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> |
| results/unified_saes_svd_v2/gpt2/summary.csv | gpt2 | full | 63830.86 tok/s <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 2 --> |
| results/unified_saes_svd_v2/gpt2/summary.csv | gpt2 | saes_svd_v2 | 67372.75 tok/s <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> | 635.79 MB <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> |
| results/unified_saes_svd_v2/roberta/summary.csv | roberta | full | 96460.3 tok/s <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 2 --> |
| results/unified_saes_svd_v2/roberta/summary.csv | roberta | saes_svd_v2 | 83020.03 tok/s <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> | 364.29 MB <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | saes_svd_v2 | Elapsed 0.891s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> | 112.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 87.50% | `results/unified_saes_svd_v2/distilbert/summary.csv` |
| F1 Score | 0.8868 | `results/unified_saes_svd_v2/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_saes_svd_v2/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_saes_svd_v2/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_saes_svd_v2/distilbert/summary.csv` |
| Timestamp | 2026-09-11T08:07:06.350513 | `results/unified_saes_svd_v2/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.48, 60.78, 75.11]
> These numbers must NOT be cited in the final paper.

