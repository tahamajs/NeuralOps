| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Slicegpt** | **80.62%** | -9.86% | PASS | 0.8312 | 0 | **1.00x** | 339.7 MB | 19.3 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Slicegpt** | **50.92%** | -24.19% | PASS_LOW | 0.6748 | 0 | **1.00x** | 366.8 MB | 25.8 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Slicegpt** | **59.06%** | -1.72% | PASS | 0.6270 | 0 | **1.00x** | 637.2 MB | 24.6 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_slicegpt/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> |
| results/unified_slicegpt/distilbert/summary.csv | distilbert | slicegpt | PASS | 80.62% <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> | 0.8312 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> |
| results/unified_slicegpt/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> |
| results/unified_slicegpt/gpt2/summary.csv | gpt2 | slicegpt | PASS | 59.06% <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> | 0.627 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> |
| results/unified_slicegpt/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> |
| results/unified_slicegpt/roberta/summary.csv | roberta | slicegpt | PASS_LOW | 50.92% <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slicegpt | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_slicegpt/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> |
| results/unified_slicegpt/distilbert/summary.csv | distilbert | slicegpt | 66955010 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> | 4135682 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> |
| results/unified_slicegpt/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> |
| results/unified_slicegpt/gpt2/summary.csv | gpt2 | slicegpt | 124441344 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> | 1536 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> |
| results/unified_slicegpt/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> |
| results/unified_slicegpt/roberta/summary.csv | roberta | slicegpt | 124647170 <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> | 7679234 <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slicegpt | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_slicegpt/distilbert/summary.csv | distilbert | full | 141027.67 tok/s <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 2 --> |
| results/unified_slicegpt/distilbert/summary.csv | distilbert | slicegpt | 126869.88 tok/s <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> | 339.71 MB <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> | 0.4834 <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> |
| results/unified_slicegpt/gpt2/summary.csv | gpt2 | full | 63519.12 tok/s <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 2 --> |
| results/unified_slicegpt/gpt2/summary.csv | gpt2 | slicegpt | 62933.0 tok/s <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> | 637.16 MB <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> | 0.4988 <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> |
| results/unified_slicegpt/roberta/summary.csv | roberta | full | 93726.44 tok/s <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_slicegpt/roberta/summary.csv line 2 --> |
| results/unified_slicegpt/roberta/summary.csv | roberta | slicegpt | 84266.24 tok/s <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> | 366.79 MB <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> | 0.4892 <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | slicegpt | Elapsed 0.284s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> | 207.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 86.00% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.8531 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T18:35:28.823958 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.48, 75.11]
> These numbers must NOT be cited in the final paper.

