# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_lsc/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> |
| results/unified_lsc/distilbert/summary.csv | distilbert | lsc | PASS | 90.48% <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> | 0.9077 <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> |
| results/unified_lsc/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> |
| results/unified_lsc/gpt2/summary.csv | gpt2 | lsc | PASS | 62.04% <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> | 0.6783 <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> |
| results/unified_lsc/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> |
| results/unified_lsc/roberta/summary.csv | roberta | lsc | PASS | 72.02% <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> | 0.7724 <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lsc | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lsc_compressor | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_lsc/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> |
| results/unified_lsc/distilbert/summary.csv | distilbert | lsc | 66955010 <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> |
| results/unified_lsc/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> |
| results/unified_lsc/gpt2/summary.csv | gpt2 | lsc | 124441344 <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> |
| results/unified_lsc/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> |
| results/unified_lsc/roberta/summary.csv | roberta | lsc | 124647170 <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lsc | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lsc_compressor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_lsc/distilbert/summary.csv | distilbert | full | 170128.55 tok/s <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lsc/distilbert/summary.csv line 2 --> |
| results/unified_lsc/distilbert/summary.csv | distilbert | lsc | 184866.23 tok/s <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> |
| results/unified_lsc/gpt2/summary.csv | gpt2 | full | 69085.06 tok/s <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lsc/gpt2/summary.csv line 2 --> |
| results/unified_lsc/gpt2/summary.csv | gpt2 | lsc | 68032.35 tok/s <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> |
| results/unified_lsc/roberta/summary.csv | roberta | full | 104692.96 tok/s <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_lsc/roberta/summary.csv line 2 --> |
| results/unified_lsc/roberta/summary.csv | roberta | lsc | 105976.8 tok/s <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lsc | Elapsed 0.402s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> | 197.3 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | lsc_compressor | Elapsed 0.189s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> | 197.3 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> |

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
> [90.48, 90.48, 60.78, 62.04, 75.11, 72.02]
> These numbers must NOT be cited in the final paper.

