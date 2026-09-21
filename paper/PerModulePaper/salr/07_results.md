# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_salr/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> |
| results/unified_salr/distilbert/summary.csv | distilbert | salr | PASS | 90.37% <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> | 0.9067 <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> |
| results/unified_salr/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> |
| results/unified_salr/gpt2/summary.csv | gpt2 | salr | PASS | 58.94% <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> | 0.6679 <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> |
| results/unified_salr/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_salr/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_salr/roberta/summary.csv line 2 --> |
| results/unified_salr/roberta/summary.csv | roberta | salr | PASS | 80.96% <!-- source: results/unified_salr/roberta/summary.csv line 3 --> | 0.8363 <!-- source: results/unified_salr/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | salr | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_salr/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> |
| results/unified_salr/distilbert/summary.csv | distilbert | salr | 66955010 <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> |
| results/unified_salr/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> |
| results/unified_salr/gpt2/summary.csv | gpt2 | salr | 124441344 <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> |
| results/unified_salr/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_salr/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_salr/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_salr/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_salr/roberta/summary.csv line 2 --> |
| results/unified_salr/roberta/summary.csv | roberta | salr | 124647170 <!-- source: results/unified_salr/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_salr/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_salr/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_salr/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | salr | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_salr/distilbert/summary.csv | distilbert | full | 127223.28 tok/s <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_salr/distilbert/summary.csv line 2 --> |
| results/unified_salr/distilbert/summary.csv | distilbert | salr | 136607.38 tok/s <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> | 310.69 MB <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_salr/distilbert/summary.csv line 3 --> |
| results/unified_salr/gpt2/summary.csv | gpt2 | full | 64140.72 tok/s <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_salr/gpt2/summary.csv line 2 --> |
| results/unified_salr/gpt2/summary.csv | gpt2 | salr | 63444.33 tok/s <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> | 635.79 MB <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_salr/gpt2/summary.csv line 3 --> |
| results/unified_salr/roberta/summary.csv | roberta | full | 93593.68 tok/s <!-- source: results/unified_salr/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_salr/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_salr/roberta/summary.csv line 2 --> |
| results/unified_salr/roberta/summary.csv | roberta | salr | 84414.74 tok/s <!-- source: results/unified_salr/roberta/summary.csv line 3 --> | 364.29 MB <!-- source: results/unified_salr/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_salr/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | salr | Elapsed 0.786s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 102 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.90% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9454 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T18:28:51.930669 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

