# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_uniql/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> |
| results/unified_uniql/distilbert/summary.csv | distilbert | uniql | PASS | 90.14% <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> | 0.9027 <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> |
| results/unified_uniql/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> |
| results/unified_uniql/gpt2/summary.csv | gpt2 | uniql | PASS | 60.89% <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> | 0.6634 <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> |
| results/unified_uniql/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> |
| results/unified_uniql/roberta/summary.csv | roberta | uniql | PASS | 82.45% <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> | 0.8468 <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | uniql | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | uniql_compressor | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_uniql/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> |
| results/unified_uniql/distilbert/summary.csv | distilbert | uniql | 66955010 <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> |
| results/unified_uniql/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> |
| results/unified_uniql/gpt2/summary.csv | gpt2 | uniql | 124441344 <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> |
| results/unified_uniql/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> |
| results/unified_uniql/roberta/summary.csv | roberta | uniql | 124647170 <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | uniql | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | uniql_compressor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_uniql/distilbert/summary.csv | distilbert | full | 157493.47 tok/s <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_uniql/distilbert/summary.csv line 2 --> |
| results/unified_uniql/distilbert/summary.csv | distilbert | uniql | 150882.16 tok/s <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> | 339.22 MB <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> |
| results/unified_uniql/gpt2/summary.csv | gpt2 | full | 68723.9 tok/s <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_uniql/gpt2/summary.csv line 2 --> |
| results/unified_uniql/gpt2/summary.csv | gpt2 | uniql | 72184.34 tok/s <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> | 635.04 MB <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> |
| results/unified_uniql/roberta/summary.csv | roberta | full | 99725.16 tok/s <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_uniql/roberta/summary.csv line 2 --> |
| results/unified_uniql/roberta/summary.csv | roberta | uniql | 104803.31 tok/s <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> | 359.04 MB <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | uniql | Elapsed 0.813s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> | 128.5 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | uniql_compressor | Elapsed 0.81s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> | 127.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> |

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
> [90.48, 90.14, 60.78, 60.89, 75.11, 82.45]
> These numbers must NOT be cited in the final paper.

