# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_cur/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> |
| results/unified_cur/distilbert/summary.csv | distilbert | cur | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> |
| results/unified_cur/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> |
| results/unified_cur/gpt2/summary.csv | gpt2 | cur | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> |
| results/unified_cur/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_cur/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_cur/roberta/summary.csv line 2 --> |
| results/unified_cur/roberta/summary.csv | roberta | cur | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_cur/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_cur/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cur | PASS | Comp 30.75x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_cur/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> |
| results/unified_cur/distilbert/summary.csv | distilbert | cur | 66955010 <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> |
| results/unified_cur/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> |
| results/unified_cur/gpt2/summary.csv | gpt2 | cur | 124441344 <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> |
| results/unified_cur/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_cur/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_cur/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_cur/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cur/roberta/summary.csv line 2 --> |
| results/unified_cur/roberta/summary.csv | roberta | cur | 124647170 <!-- source: results/unified_cur/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_cur/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_cur/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cur/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cur | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> | Reduced (30.75x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> | 30.75x <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_cur/distilbert/summary.csv | distilbert | full | 48999.55 tok/s <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur/distilbert/summary.csv line 2 --> |
| results/unified_cur/distilbert/summary.csv | distilbert | cur | 52088.0 tok/s <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> |
| results/unified_cur/gpt2/summary.csv | gpt2 | full | 28102.48 tok/s <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur/gpt2/summary.csv line 2 --> |
| results/unified_cur/gpt2/summary.csv | gpt2 | cur | 25444.96 tok/s <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> | 627.28 MB <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> |
| results/unified_cur/roberta/summary.csv | roberta | full | 27142.67 tok/s <!-- source: results/unified_cur/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_cur/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cur/roberta/summary.csv line 2 --> |
| results/unified_cur/roberta/summary.csv | roberta | cur | 26930.36 tok/s <!-- source: results/unified_cur/roberta/summary.csv line 3 --> | 519.68 MB <!-- source: results/unified_cur/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_cur/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cur | Elapsed 1.041s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> | 106.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.70% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.9433 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:15:40.938394 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |

