# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_slim/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> |
| results/unified_slim/distilbert/summary.csv | distilbert | slim | PASS | 87.96% <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> | 0.8889 <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> |
| results/unified_slim/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> |
| results/unified_slim/gpt2/summary.csv | gpt2 | slim | PASS_LOW | 52.41% <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> | 0.6765 <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> |
| results/unified_slim/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_slim/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_slim/roberta/summary.csv line 2 --> |
| results/unified_slim/roberta/summary.csv | roberta | slim | PASS_LOW | 50.92% <!-- source: results/unified_slim/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_slim/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slim | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slim_compressor | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_slim/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> |
| results/unified_slim/distilbert/summary.csv | distilbert | slim | 66955010 <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> |
| results/unified_slim/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> |
| results/unified_slim/gpt2/summary.csv | gpt2 | slim | 124441344 <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> |
| results/unified_slim/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_slim/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_slim/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_slim/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slim/roberta/summary.csv line 2 --> |
| results/unified_slim/roberta/summary.csv | roberta | slim | 124647170 <!-- source: results/unified_slim/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_slim/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_slim/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slim/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slim | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slim_compressor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_slim/distilbert/summary.csv | distilbert | full | 135680.86 tok/s <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slim/distilbert/summary.csv line 2 --> |
| results/unified_slim/distilbert/summary.csv | distilbert | slim | 146796.93 tok/s <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> | 339.21 MB <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> | 0.5282 <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> |
| results/unified_slim/gpt2/summary.csv | gpt2 | full | 65710.32 tok/s <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slim/gpt2/summary.csv line 2 --> |
| results/unified_slim/gpt2/summary.csv | gpt2 | slim | 64458.1 tok/s <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> | 640.66 MB <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> | 0.692 <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> |
| results/unified_slim/roberta/summary.csv | roberta | full | 91493.48 tok/s <!-- source: results/unified_slim/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_slim/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slim/roberta/summary.csv line 2 --> |
| results/unified_slim/roberta/summary.csv | roberta | slim | 82511.43 tok/s <!-- source: results/unified_slim/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_slim/roberta/summary.csv line 3 --> | 0.4214 <!-- source: results/unified_slim/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slim | Elapsed 1.028s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> | 196.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slim_compressor | Elapsed 0.781s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> | 196.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 0.00% | `results/all_121_compressors_gpu_matrix.csv` |
| F1 Score | 0.0000 | `results/all_121_compressors_gpu_matrix.csv` |
| Compression | 1.000x | `results/all_121_compressors_gpu_matrix.csv` |
| Compressed Layers | ? | `results/all_121_compressors_gpu_matrix.csv` |
| Fallback Layers | ? | `results/all_121_compressors_gpu_matrix.csv` |
| Timestamp | unknown | `results/all_121_compressors_gpu_matrix.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.48, 87.96, 60.78, 52.41, 75.11, 50.92]
> These numbers must NOT be cited in the final paper.

