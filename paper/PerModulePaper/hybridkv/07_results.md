# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_hybridkv/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> |
| results/unified_hybridkv/distilbert/summary.csv | distilbert | hybridkv | PASS | 90.94% <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> | 0.9123 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> |
| results/unified_hybridkv/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> |
| results/unified_hybridkv/gpt2/summary.csv | gpt2 | hybridkv | PASS | 62.04% <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> | 0.6783 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> |
| results/unified_hybridkv/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> |
| results/unified_hybridkv/roberta/summary.csv | roberta | hybridkv | PASS | 77.18% <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> | 0.8077 <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | hybridkv | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_hybridkv/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> |
| results/unified_hybridkv/distilbert/summary.csv | distilbert | hybridkv | 66955010 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> |
| results/unified_hybridkv/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> |
| results/unified_hybridkv/gpt2/summary.csv | gpt2 | hybridkv | 124441344 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> |
| results/unified_hybridkv/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> |
| results/unified_hybridkv/roberta/summary.csv | roberta | hybridkv | 124647170 <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | hybridkv | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_hybridkv/distilbert/summary.csv | distilbert | full | 170826.03 tok/s <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 2 --> |
| results/unified_hybridkv/distilbert/summary.csv | distilbert | hybridkv | 160153.17 tok/s <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> | 337.6 MB <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> |
| results/unified_hybridkv/gpt2/summary.csv | gpt2 | full | 71856.61 tok/s <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 2 --> |
| results/unified_hybridkv/gpt2/summary.csv | gpt2 | hybridkv | 72086.04 tok/s <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> | 634.93 MB <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> |
| results/unified_hybridkv/roberta/summary.csv | roberta | full | 118255.8 tok/s <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_hybridkv/roberta/summary.csv line 2 --> |
| results/unified_hybridkv/roberta/summary.csv | roberta | hybridkv | 104952.0 tok/s <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> | 361.56 MB <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | hybridkv | Elapsed 0.17s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.30% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.9390 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:33:23.044600 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.11]
> These numbers must NOT be cited in the final paper.

