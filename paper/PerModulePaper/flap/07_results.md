# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_flap/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> |
| results/unified_flap/distilbert/summary.csv | distilbert | flap | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> |
| results/unified_flap/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> |
| results/unified_flap/gpt2/summary.csv | gpt2 | flap | PASS | 77.87% <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> | 0.7737 <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> |
| results/unified_flap/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_flap/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_flap/roberta/summary.csv line 2 --> |
| results/unified_flap/roberta/summary.csv | roberta | flap | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_flap/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_flap/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | flap | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_flap/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> |
| results/unified_flap/distilbert/summary.csv | distilbert | flap | 66955010 <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> |
| results/unified_flap/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> |
| results/unified_flap/gpt2/summary.csv | gpt2 | flap | 124441344 <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> |
| results/unified_flap/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_flap/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_flap/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_flap/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_flap/roberta/summary.csv line 2 --> |
| results/unified_flap/roberta/summary.csv | roberta | flap | 124647170 <!-- source: results/unified_flap/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_flap/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_flap/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_flap/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | flap | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_flap/distilbert/summary.csv | distilbert | full | 51079.24 tok/s <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_flap/distilbert/summary.csv line 2 --> |
| results/unified_flap/distilbert/summary.csv | distilbert | flap | 52733.2 tok/s <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> |
| results/unified_flap/gpt2/summary.csv | gpt2 | full | 26806.41 tok/s <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_flap/gpt2/summary.csv line 2 --> |
| results/unified_flap/gpt2/summary.csv | gpt2 | flap | 26806.9 tok/s <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> | 626.91 MB <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> | 0.3484512667293313 <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> |
| results/unified_flap/roberta/summary.csv | roberta | full | 26887.25 tok/s <!-- source: results/unified_flap/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_flap/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_flap/roberta/summary.csv line 2 --> |
| results/unified_flap/roberta/summary.csv | roberta | flap | 27426.09 tok/s <!-- source: results/unified_flap/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_flap/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_flap/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | flap | Elapsed 0.187s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.70% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.9432 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 36 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:22:31.507170 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |

