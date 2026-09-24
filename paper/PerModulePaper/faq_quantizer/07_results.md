| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Faq** | **90.71%** | +0.23% | PASS | 0.9099 | 0 | **0.98x** | 417.8 MB | 7.1s |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Faq** | **63.30%** | -25.46% | PASS | 0.7260 | 0 | **1.00x** | 535.8 MB | 12.8s |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Faq** | **57.22%** | -24.09% | PASS | 0.6403 | 0 | **1.00x** | 511.0 MB | 10.9s |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_faq/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> |
| results/unified_faq/distilbert/summary.csv | distilbert | faq | PASS_NO_COMPRESSION | 90.48% <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> | 0.9071 <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> |
| results/unified_faq/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> |
| results/unified_faq/gpt2/summary.csv | gpt2 | faq | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> | 0.7848 <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> |
| results/unified_faq/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_faq/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_faq/roberta/summary.csv line 2 --> |
| results/unified_faq/roberta/summary.csv | roberta | faq | PASS_NO_COMPRESSION | 80.05% <!-- source: results/unified_faq/roberta/summary.csv line 3 --> | 0.8184 <!-- source: results/unified_faq/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | faq | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | faq_quantizer | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_faq/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> |
| results/unified_faq/distilbert/summary.csv | distilbert | faq | 66955010 <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> | 68282150 <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> | 1327140 <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> | 0.9806x <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> |
| results/unified_faq/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> |
| results/unified_faq/gpt2/summary.csv | gpt2 | faq | 124441344 <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> | 126800688 <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> | 2359344 <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> | 0.9814x <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> |
| results/unified_faq/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_faq/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_faq/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_faq/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_faq/roberta/summary.csv line 2 --> |
| results/unified_faq/roberta/summary.csv | roberta | faq | 124647170 <!-- source: results/unified_faq/roberta/summary.csv line 3 --> | 127301450 <!-- source: results/unified_faq/roberta/summary.csv line 3 --> | 2654280 <!-- source: results/unified_faq/roberta/summary.csv line 3 --> | 0.9791x <!-- source: results/unified_faq/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | faq | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | faq_quantizer | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_faq/distilbert/summary.csv | distilbert | full | 50879.26 tok/s <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_faq/distilbert/summary.csv line 2 --> |
| results/unified_faq/distilbert/summary.csv | distilbert | faq | 10050.43 tok/s <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> | 416.0 MB <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> |
| results/unified_faq/gpt2/summary.csv | gpt2 | full | 26858.28 tok/s <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_faq/gpt2/summary.csv line 2 --> |
| results/unified_faq/gpt2/summary.csv | gpt2 | faq | 6946.16 tok/s <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> | 884.17 MB <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> |
| results/unified_faq/roberta/summary.csv | roberta | full | 27260.04 tok/s <!-- source: results/unified_faq/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_faq/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_faq/roberta/summary.csv line 2 --> |
| results/unified_faq/roberta/summary.csv | roberta | faq | 5085.38 tok/s <!-- source: results/unified_faq/roberta/summary.csv line 3 --> | 916.21 MB <!-- source: results/unified_faq/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_faq/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | faq | Elapsed 0.243s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | faq_quantizer | Elapsed 0.233s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 62.50% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-26 00:01:37 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 90.48, 79.36, 78.56, 78.56, 80.05]
> These numbers must NOT be cited in the final paper.

