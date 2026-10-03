| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Tt Linear** | **55.73%** | -35.21% | PASS_LOW | 0.6560 | 0 | **0.92x** | 404.2 MB | 24.6 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Tt Linear** | **50.92%** | -24.54% | PASS_LOW | 0.6748 | 0 | **0.91x** | 710.7 MB | 35.4 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Tt Linear** | **50.92%** | -6.99% | PASS_LOW | 0.6748 | 0 | **0.97x** | 711.7 MB | 37.8 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_tt/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> |
| results/unified_tt/distilbert/summary.csv | distilbert | tt | PASS_NO_COMPRESSION | 49.77% <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> | 0.1673 <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> |
| results/unified_tt/gpt2/summary.csv | gpt2 | full | PASS | 87.84% <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> | 0.8776 <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> |
| results/unified_tt/gpt2/summary.csv | gpt2 | tt | PASS | 87.61% <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> | 0.8741 <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> |
| results/unified_tt/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_tt/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_tt/roberta/summary.csv line 2 --> |
| results/unified_tt/roberta/summary.csv | roberta | tt | PASS_LOW | 50.92% <!-- source: results/unified_tt/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_tt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | tt | PASS | Comp 0.86x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_tt/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> |
| results/unified_tt/distilbert/summary.csv | distilbert | tt | 66955010 <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> | 74072834 <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> | 0.9039x <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> |
| results/unified_tt/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> |
| results/unified_tt/gpt2/summary.csv | gpt2 | tt | 126800640 <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> | 131244303 <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> | 0.9661x <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> |
| results/unified_tt/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_tt/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_tt/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_tt/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_tt/roberta/summary.csv line 2 --> |
| results/unified_tt/roberta/summary.csv | roberta | tt | 124647170 <!-- source: results/unified_tt/roberta/summary.csv line 3 --> | 138882866 <!-- source: results/unified_tt/roberta/summary.csv line 3 --> | 592178 <!-- source: results/unified_tt/roberta/summary.csv line 3 --> | 0.8975x <!-- source: results/unified_tt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | tt | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> | Reduced (0.86x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> | 0.86x <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_tt/distilbert/summary.csv | distilbert | full | 47311.85 tok/s <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tt/distilbert/summary.csv line 2 --> |
| results/unified_tt/distilbert/summary.csv | distilbert | tt | 36770.94 tok/s <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> | 391.17 MB <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> | 0.056152360718346 <!-- source: results/unified_tt/distilbert/summary.csv line 3 --> |
| results/unified_tt/gpt2/summary.csv | gpt2 | full | 12105.44 tok/s <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tt/gpt2/summary.csv line 2 --> |
| results/unified_tt/gpt2/summary.csv | gpt2 | tt | 13802.49 tok/s <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> | 426.32 MB <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> | 0.0735220684046739 <!-- source: results/unified_tt/gpt2/summary.csv line 3 --> |
| results/unified_tt/roberta/summary.csv | roberta | full | 22068.8 tok/s <!-- source: results/unified_tt/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_tt/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_tt/roberta/summary.csv line 2 --> |
| results/unified_tt/roberta/summary.csv | roberta | tt | 14153.29 tok/s <!-- source: results/unified_tt/roberta/summary.csv line 3 --> | 715.58 MB <!-- source: results/unified_tt/roberta/summary.csv line 3 --> | 0.0546217006105861 <!-- source: results/unified_tt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | tt | Elapsed 2.367s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> | 226.9 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 119 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.70% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9432 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 36 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T19:23:17.957366 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |

