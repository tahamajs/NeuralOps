| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Grecodec** | **90.37%** | -0.11% | PASS | 0.9069 | 0 | **1.00x** | 310.7 MB | 18.3 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Grecodec** | **88.53%** | -0.23% | PASS | 0.8936 | 0 | **1.00x** | 321.6 MB | 28.0 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Grecodec** | **82.57%** | +1.26% | PASS | 0.8380 | 0 | **1.00x** | 398.7 MB | 24.4 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_grecodec/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> |
| results/unified_grecodec/distilbert/summary.csv | distilbert | grecodec | PASS | 90.37% <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> | 0.9069 <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> |
| results/unified_grecodec/gpt2/summary.csv | gpt2 | full | PASS | 81.31% <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> | 0.8372 <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> |
| results/unified_grecodec/gpt2/summary.csv | gpt2 | grecodec | PASS | 82.57% <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> | 0.838 <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> |
| results/unified_grecodec/roberta/summary.csv | roberta | full | PASS | 88.76% <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> | 0.8942 <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> |
| results/unified_grecodec/roberta/summary.csv | roberta | grecodec | PASS | 88.53% <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> | 0.8936 <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | grecodec | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_grecodec/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> |
| results/unified_grecodec/distilbert/summary.csv | distilbert | grecodec | 66955010 <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> |
| results/unified_grecodec/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> |
| results/unified_grecodec/gpt2/summary.csv | gpt2 | grecodec | 126800640 <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> | 126800640 <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> |
| results/unified_grecodec/roberta/summary.csv | roberta | full | 127325954 <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> | 127325954 <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> |
| results/unified_grecodec/roberta/summary.csv | roberta | grecodec | 127325954 <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> | 127325954 <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> | 616706 <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | grecodec | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_grecodec/distilbert/summary.csv | distilbert | full | 164004.33 tok/s <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_grecodec/distilbert/summary.csv line 2 --> |
| results/unified_grecodec/distilbert/summary.csv | distilbert | grecodec | 173191.15 tok/s <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> | 310.69 MB <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> | 0.661879214975569 <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> |
| results/unified_grecodec/gpt2/summary.csv | gpt2 | full | 85994.13 tok/s <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_grecodec/gpt2/summary.csv line 2 --> |
| results/unified_grecodec/gpt2/summary.csv | gpt2 | grecodec | 60393.06 tok/s <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> | 398.74 MB <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> | 0.7040223963558674 <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> |
| results/unified_grecodec/roberta/summary.csv | roberta | full | 114679.6 tok/s <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> | 299.92 MB <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_grecodec/roberta/summary.csv line 2 --> |
| results/unified_grecodec/roberta/summary.csv | roberta | grecodec | 42992.33 tok/s <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> | 321.64 MB <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> | 0.667251886592971 <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | grecodec | Elapsed 6.41s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> | 101.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.80% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.9444 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:27:53.797656 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |

