# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_nli_softmax/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> |
| results/unified_nli_softmax/distilbert/summary.csv | distilbert | nli_softmax | PASS | 89.33% <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> | 0.8905 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> |
| results/unified_nli_softmax/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> |
| results/unified_nli_softmax/gpt2/summary.csv | gpt2 | nli_softmax | PASS_LOW | 55.28% <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> | 0.6782 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> |
| results/unified_nli_softmax/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> |
| results/unified_nli_softmax/roberta/summary.csv | roberta | nli_softmax | PASS | 60.32% <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> | 0.482 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nli_softmax | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_nli_softmax/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> |
| results/unified_nli_softmax/distilbert/summary.csv | distilbert | nli_softmax | 66955010 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> | 66955016 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> | 18438 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> |
| results/unified_nli_softmax/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> |
| results/unified_nli_softmax/gpt2/summary.csv | gpt2 | nli_softmax | 124441344 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> | 124441356 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> | 36876 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> |
| results/unified_nli_softmax/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> |
| results/unified_nli_softmax/roberta/summary.csv | roberta | nli_softmax | 124647170 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> | 131734286 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> | 36876 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nli_softmax | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_nli_softmax/distilbert/summary.csv | distilbert | full | 51810.89 tok/s <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 2 --> |
| results/unified_nli_softmax/distilbert/summary.csv | distilbert | nli_softmax | 42469.18 tok/s <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> | 315.91 MB <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> |
| results/unified_nli_softmax/gpt2/summary.csv | gpt2 | full | 26180.39 tok/s <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 2 --> |
| results/unified_nli_softmax/gpt2/summary.csv | gpt2 | nli_softmax | 20249.29 tok/s <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> | 557.88 MB <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> |
| results/unified_nli_softmax/roberta/summary.csv | roberta | full | 26692.46 tok/s <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 2 --> |
| results/unified_nli_softmax/roberta/summary.csv | roberta | nli_softmax | 21556.3 tok/s <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> | 640.61 MB <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | nli_softmax | Elapsed 0.166s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.14% | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| F1 Score | 0.8998 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |
| Timestamp | 2026-09-19T01:38:36.308868 | `results/unified_sweep/distilbert/summary_buggy_backup.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [79.36, 78.56]
> These numbers must NOT be cited in the final paper.

