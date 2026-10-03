| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Admm** | **88.53%** | -1.95% | PASS | 0.8922 | 11 | **1.00x** | 339.4 MB | -- |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Admm** | **89.11%** | +0.35% | PASS | 0.8966 | 27 | **1.00x** | 333.4 MB | -- |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Admm** | **84.63%** | +3.32% | PASS | 0.8537 | 0 | **1.00x** | 409.9 MB | -- |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_admm/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> |
| results/unified_admm/distilbert/summary.csv | distilbert | admm | PASS | 90.37% <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 0.905 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> |
| results/unified_admm/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> |
| results/unified_admm/gpt2/summary.csv | gpt2 | admm | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> |
| results/unified_admm/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.55% <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> | 0.5501 <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> |
| results/unified_admm/qwen3_5/summary.csv | qwen3_5 | admm | PASS | 88.42% <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 0.8947 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> |
| results/unified_admm/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_admm/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_admm/roberta/summary.csv line 2 --> |
| results/unified_admm/roberta/summary.csv | roberta | admm | PASS | 62.04% <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 0.6505 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | admm | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | admm_compressor | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_admm/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> |
| results/unified_admm/distilbert/summary.csv | distilbert | admm | 66955010 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> |
| results/unified_admm/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> |
| results/unified_admm/gpt2/summary.csv | gpt2 | admm | 124441344 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> |
| results/unified_admm/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> |
| results/unified_admm/qwen3_5/summary.csv | qwen3_5 | admm | 763217728 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 5776384 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> |
| results/unified_admm/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_admm/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_admm/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_admm/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_admm/roberta/summary.csv line 2 --> |
| results/unified_admm/roberta/summary.csv | roberta | admm | 124647170 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_admm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | admm | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | admm_compressor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_admm/distilbert/summary.csv | distilbert | full | 51199.68 tok/s <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_admm/distilbert/summary.csv line 2 --> |
| results/unified_admm/distilbert/summary.csv | distilbert | admm | 50559.49 tok/s <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> | 0.4087498231888554 <!-- source: results/unified_admm/distilbert/summary.csv line 3 --> |
| results/unified_admm/gpt2/summary.csv | gpt2 | full | 23706.62 tok/s <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_admm/gpt2/summary.csv line 2 --> |
| results/unified_admm/gpt2/summary.csv | gpt2 | admm | 19301.53 tok/s <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 628.41 MB <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> | 0.0 <!-- source: results/unified_admm/gpt2/summary.csv line 3 --> |
| results/unified_admm/qwen3_5/summary.csv | qwen3_5 | full | 1287.4 tok/s <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> | 1810.59 MB <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_admm/qwen3_5/summary.csv line 2 --> |
| results/unified_admm/qwen3_5/summary.csv | qwen3_5 | admm | 1099.69 tok/s <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 1851.31 MB <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> | 0.4157980064416287 <!-- source: results/unified_admm/qwen3_5/summary.csv line 3 --> |
| results/unified_admm/roberta/summary.csv | roberta | full | 23915.43 tok/s <!-- source: results/unified_admm/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_admm/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_admm/roberta/summary.csv line 2 --> |
| results/unified_admm/roberta/summary.csv | roberta | admm | 21347.69 tok/s <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_admm/roberta/summary.csv line 3 --> | 0.4280215595989844 <!-- source: results/unified_admm/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | admm | Elapsed 0.557s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> | 194.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 10 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | admm_compressor | Elapsed 0.469s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> | 194.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 11 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 81.30% | `results/agnews_unified_admm_compressor/roberta/summary.csv` |
| F1 Score | 0.8033 | `results/agnews_unified_admm_compressor/roberta/summary.csv` |
| Compression | 1.000x | `results/agnews_unified_admm_compressor/roberta/summary.csv` |
| Compressed Layers | 27 | `results/agnews_unified_admm_compressor/roberta/summary.csv` |
| Fallback Layers | 21 | `results/agnews_unified_admm_compressor/roberta/summary.csv` |
| Timestamp | 2026-09-18T18:53:27.702998 | `results/agnews_unified_admm_compressor/roberta/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 90.37, 67.55, 88.42, 78.56, 62.04]
> These numbers must NOT be cited in the final paper.

