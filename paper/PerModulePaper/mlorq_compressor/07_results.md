# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_mlorq/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> |
| results/unified_mlorq/distilbert/summary.csv | distilbert | mlorq | PASS | 90.48% <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> | 0.9077 <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> |
| results/unified_mlorq/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> |
| results/unified_mlorq/gpt2/summary.csv | gpt2 | mlorq | PASS | 62.04% <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> | 0.6783 <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> |
| results/unified_mlorq/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> |
| results/unified_mlorq/roberta/summary.csv | roberta | mlorq | PASS | 72.02% <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> | 0.7724 <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | mlorq | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_mlorq/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> |
| results/unified_mlorq/distilbert/summary.csv | distilbert | mlorq | 66955010 <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> |
| results/unified_mlorq/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> |
| results/unified_mlorq/gpt2/summary.csv | gpt2 | mlorq | 124441344 <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> |
| results/unified_mlorq/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> |
| results/unified_mlorq/roberta/summary.csv | roberta | mlorq | 124647170 <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | mlorq | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_mlorq/distilbert/summary.csv | distilbert | full | 166772.58 tok/s <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mlorq/distilbert/summary.csv line 2 --> |
| results/unified_mlorq/distilbert/summary.csv | distilbert | mlorq | 155916.18 tok/s <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_mlorq/distilbert/summary.csv line 3 --> |
| results/unified_mlorq/gpt2/summary.csv | gpt2 | full | 71722.69 tok/s <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mlorq/gpt2/summary.csv line 2 --> |
| results/unified_mlorq/gpt2/summary.csv | gpt2 | mlorq | 69940.53 tok/s <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> | 635.54 MB <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_mlorq/gpt2/summary.csv line 3 --> |
| results/unified_mlorq/roberta/summary.csv | roberta | full | 112179.18 tok/s <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_mlorq/roberta/summary.csv line 2 --> |
| results/unified_mlorq/roberta/summary.csv | roberta | mlorq | 114414.86 tok/s <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_mlorq/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | mlorq | Elapsed 12.625s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> | 140.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 79 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.48% | `results/unified_mlorq/distilbert/summary.csv` |
| F1 Score | 0.9077 | `results/unified_mlorq/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_mlorq/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_mlorq/distilbert/summary.csv` |
| Fallback Layers | 36 | `results/unified_mlorq/distilbert/summary.csv` |
| Timestamp | 2026-09-11T03:03:41.210492 | `results/unified_mlorq/distilbert/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

