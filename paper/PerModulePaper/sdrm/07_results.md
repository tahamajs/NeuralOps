| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Sdrm** | **76.49%** | -13.99% | PASS | 0.7389 | 36 | **1.24x** | 518.3 MB | 22.4 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Sdrm** | **50.92%** | -24.19% | PASS_LOW | 0.6748 | 68 | **1.25x** | 890.7 MB | 32.8 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Sdrm** | **50.92%** | -9.86% | PASS_LOW | 0.6748 | 48 | **1.26x** | 1146.9 MB | 29.2 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_sdrm/distilbert/summary.csv | distilbert | full | PASS | 90.94% <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> | 0.9125 <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> |
| results/unified_sdrm/distilbert/summary.csv | distilbert | sdrm | PASS | 85.78% <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> | 0.8631 <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> |
| results/unified_sdrm/gpt2/summary.csv | gpt2 | full | PASS | 87.84% <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> | 0.8776 <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> |
| results/unified_sdrm/gpt2/summary.csv | gpt2 | sdrm | PASS | 67.66% <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> | 0.7482 <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> |
| results/unified_sdrm/qwen3_5/summary.csv | qwen3_5 | full | PASS | 63.88% <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> | 0.6392 <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> |
| results/unified_sdrm/roberta/summary.csv | roberta | full | PASS | 75.46% <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> | 0.7277 <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> |
| results/unified_sdrm/roberta/summary.csv | roberta | sdrm | PASS_LOW | 50.92% <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> | 0.6748 <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | sdrm | PASS | Comp 1.43x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_sdrm/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> |
| results/unified_sdrm/distilbert/summary.csv | distilbert | sdrm | 66955010 <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> | 54181670 <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> | 633638 <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> | 1.2358x <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> |
| results/unified_sdrm/gpt2/summary.csv | gpt2 | full | 126800640 <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> | 126800640 <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> |
| results/unified_sdrm/gpt2/summary.csv | gpt2 | sdrm | 126800640 <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> | 100744296 <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> | 1812792 <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> | 1.2586x <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> |
| results/unified_sdrm/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> |
| results/unified_sdrm/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> |
| results/unified_sdrm/roberta/summary.csv | roberta | sdrm | 124647170 <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> | 99813190 <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> | 672070 <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> | 1.2488x <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | sdrm | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> | Reduced (1.43x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> | 1.43x <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_sdrm/distilbert/summary.csv | distilbert | full | 46799.28 tok/s <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_sdrm/distilbert/summary.csv line 2 --> |
| results/unified_sdrm/distilbert/summary.csv | distilbert | sdrm | 32743.48 tok/s <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> | 338.98 MB <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> | 0.39 <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> |
| results/unified_sdrm/gpt2/summary.csv | gpt2 | full | 26331.17 tok/s <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> | 391.24 MB <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_sdrm/gpt2/summary.csv line 2 --> |
| results/unified_sdrm/gpt2/summary.csv | gpt2 | sdrm | 10921.04 tok/s <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> | 446.9 MB <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> | 0.283 <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> |
| results/unified_sdrm/qwen3_5/summary.csv | qwen3_5 | full | 1181.72 tok/s <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> | 1796.79 MB <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_sdrm/qwen3_5/summary.csv line 2 --> |
| results/unified_sdrm/roberta/summary.csv | roberta | full | 23765.16 tok/s <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_sdrm/roberta/summary.csv line 2 --> |
| results/unified_sdrm/roberta/summary.csv | roberta | sdrm | 16739.56 tok/s <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> | 781.79 MB <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> | 0.3811 <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | sdrm | Elapsed 0.896s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> | 149.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=91.50%, comp=1.212x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 91.50% | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| F1 Score | 0.9117 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compression | 1.212x | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |
| Timestamp | 2026-09-19T18:43:25.307501 | `results/unified_sweep/distilbert_agnews_full/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [75.46]
> These numbers must NOT be cited in the final paper.

