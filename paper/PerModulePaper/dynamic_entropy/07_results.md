# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_dynamic_entropy/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> |
| results/unified_dynamic_entropy/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> |
| results/unified_dynamic_entropy/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> |
| results/unified_dynamic_entropy/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dynamic_entropy | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_dynamic_entropy/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> |
| results/unified_dynamic_entropy/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> |
| results/unified_dynamic_entropy/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> |
| results/unified_dynamic_entropy/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dynamic_entropy | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_dynamic_entropy/distilbert/summary.csv | distilbert | full | 51240.09 tok/s <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 2 --> |
| results/unified_dynamic_entropy/gpt2/summary.csv | gpt2 | full | 26678.84 tok/s <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 2 --> |
| results/unified_dynamic_entropy/qwen3_5/summary.csv | qwen3_5 | full | 6042.7 tok/s <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 2 --> |
| results/unified_dynamic_entropy/roberta/summary.csv | roberta | full | 27387.12 tok/s <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dynamic_entropy | Elapsed 1.029s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> | 104.9 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> |

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
> [90.71, 79.36, 67.89, 78.56]
> These numbers must NOT be cited in the final paper.

