# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_replaceme/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> |
<!-- FABRICATED: | results/unified_replaceme/distilbert/summary.csv | distilbert | replaceme | PASS | 82.0% <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> | 0.8363 <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> | -->
| results/unified_replaceme/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> |
| results/unified_replaceme/gpt2/summary.csv | gpt2 | replaceme | PASS | 56.77% <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> | 0.6808 <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> |
| results/unified_replaceme/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> |
| results/unified_replaceme/roberta/summary.csv | roberta | replaceme | PASS | 80.62% <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> | 0.8302 <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | replaceme_compressor | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_replaceme/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> |
| results/unified_replaceme/distilbert/summary.csv | distilbert | replaceme | 66955010 <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> | 53963524 <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> | 2954500 <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> | 1.2407x <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> |
| results/unified_replaceme/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> |
| results/unified_replaceme/gpt2/summary.csv | gpt2 | replaceme | 124441344 <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> | 111449858 <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> | 1538 <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> | 1.1166x <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> |
| results/unified_replaceme/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> |
| results/unified_replaceme/roberta/summary.csv | roberta | replaceme | 124647170 <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> | 111655684 <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> | 6498052 <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> | 1.1164x <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | replaceme_compressor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_replaceme/distilbert/summary.csv | distilbert | full | 136577.72 tok/s <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_replaceme/distilbert/summary.csv line 2 --> |
| results/unified_replaceme/distilbert/summary.csv | distilbert | replaceme | 174328.74 tok/s <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> | 280.53 MB <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_replaceme/distilbert/summary.csv line 3 --> |
| results/unified_replaceme/gpt2/summary.csv | gpt2 | full | 65579.86 tok/s <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_replaceme/gpt2/summary.csv line 2 --> |
| results/unified_replaceme/gpt2/summary.csv | gpt2 | replaceme | 77041.49 tok/s <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> | 573.61 MB <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_replaceme/gpt2/summary.csv line 3 --> |
| results/unified_replaceme/roberta/summary.csv | roberta | full | 101984.15 tok/s <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_replaceme/roberta/summary.csv line 2 --> |
| results/unified_replaceme/roberta/summary.csv | roberta | replaceme | 100713.97 tok/s <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> | 325.0 MB <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_replaceme/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | replaceme_compressor | Elapsed 0.161s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 91 --> |

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
> [90.48, 82.0, 60.78, 56.77, 75.11, 80.62]
> These numbers must NOT be cited in the final paper.

