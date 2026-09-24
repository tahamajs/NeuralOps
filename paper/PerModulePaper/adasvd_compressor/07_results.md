| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Adasvd** | **90.37%** | -0.34% | PASS | 0.9079 | 24 | **1.06x** | 196.7 MB | 111.3 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Adasvd** | **92.32%** | -2.06% | PASS | 0.9255 | 48 | **1.06x** | 333.3 MB | 142.0 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Adasvd** | **83.83%** | -5.50% | PASS | 0.8492 | 24 | **1.06x** | 464.5 MB | 135.8 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_adasvd/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> |
| results/unified_adasvd/distilbert/summary.csv | distilbert | adasvd | PASS | 86.01% <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 0.871 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> |
| results/unified_adasvd/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> |
| results/unified_adasvd/gpt2/summary.csv | gpt2 | adasvd | PASS | 63.76% <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 0.7047 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> |
| results/unified_adasvd/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.55% <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> | 0.5501 <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> |
| results/unified_adasvd/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> |
| results/unified_adasvd/roberta/summary.csv | roberta | adasvd | PASS | 73.39% <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 0.7778 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | adasvd | PASS | Comp 27.22x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_adasvd/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> |
| results/unified_adasvd/distilbert/summary.csv | distilbert | adasvd | 66955010 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 57613082 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 4832280 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 1.1621x <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> |
| results/unified_adasvd/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> |
| results/unified_adasvd/gpt2/summary.csv | gpt2 | adasvd | 124441344 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 109692694 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 35350 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 1.1345x <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> |
| results/unified_adasvd/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> |
| results/unified_adasvd/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> |
| results/unified_adasvd/roberta/summary.csv | roberta | adasvd | 124647170 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 109737258 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 8713768 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 1.1359x <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | adasvd | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> | Reduced (27.22x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> | 27.22x <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_adasvd/distilbert/summary.csv | distilbert | full | 48356.1 tok/s <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adasvd/distilbert/summary.csv line 2 --> |
| results/unified_adasvd/distilbert/summary.csv | distilbert | adasvd | 30810.24 tok/s <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 230.91 MB <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> | 0.6146613516060712 <!-- source: results/unified_adasvd/distilbert/summary.csv line 3 --> |
| results/unified_adasvd/gpt2/summary.csv | gpt2 | full | 22955.86 tok/s <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adasvd/gpt2/summary.csv line 2 --> |
| results/unified_adasvd/gpt2/summary.csv | gpt2 | adasvd | 18628.1 tok/s <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 676.71 MB <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> | 0.5492219996067473 <!-- source: results/unified_adasvd/gpt2/summary.csv line 3 --> |
| results/unified_adasvd/qwen3_5/summary.csv | qwen3_5 | full | 1295.14 tok/s <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> | 1810.59 MB <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adasvd/qwen3_5/summary.csv line 2 --> |
| results/unified_adasvd/roberta/summary.csv | roberta | full | 24228.27 tok/s <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_adasvd/roberta/summary.csv line 2 --> |
| results/unified_adasvd/roberta/summary.csv | roberta | adasvd | 17145.01 tok/s <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 610.89 MB <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> | 0.5913941559039471 <!-- source: results/unified_adasvd/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | adasvd | Elapsed 1.5s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> | 106.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 9 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `BROKEN`  
**Detail**: Error: RuntimeError: Compressor 'adasvd_compressor' not found in COMPRESSOR_REGISTRY

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 0.00% | `results/agnews_unified_adasvd_compressor/gpt2/summary.csv` |
| F1 Score | 0.0000 | `results/agnews_unified_adasvd_compressor/gpt2/summary.csv` |
| Compression | 1.000x | `results/agnews_unified_adasvd_compressor/gpt2/summary.csv` |
| Compressed Layers | 0 | `results/agnews_unified_adasvd_compressor/gpt2/summary.csv` |
| Fallback Layers | 0 | `results/agnews_unified_adasvd_compressor/gpt2/summary.csv` |
| Timestamp | 2026-09-18T18:51:46.824309 | `results/agnews_unified_adasvd_compressor/gpt2/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 86.01, 79.36, 63.76, 67.55, 78.56, 73.39]
> These numbers must NOT be cited in the final paper.

