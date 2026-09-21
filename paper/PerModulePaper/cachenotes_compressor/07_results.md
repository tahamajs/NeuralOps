# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_cachenotes/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> |
<!-- FABRICATED: | results/unified_cachenotes/distilbert/summary.csv | distilbert | cachenotes | PASS | 90.6% <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> | 0.9093 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> | -->
| results/unified_cachenotes/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> |
| results/unified_cachenotes/gpt2/summary.csv | gpt2 | cachenotes | PASS | 74.77% <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> | 0.7751 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> |
| results/unified_cachenotes/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.55% <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> | 0.5501 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> |
| results/unified_cachenotes/qwen3_5/summary.csv | qwen3_5 | cachenotes | PASS | 89.22% <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> | 0.8956 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> |
| results/unified_cachenotes/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> |
| results/unified_cachenotes/roberta/summary.csv | roberta | cachenotes | PASS | 68.12% <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> | 0.7518 <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cachenotes | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cachenotes_compressor | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_cachenotes/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> |
| results/unified_cachenotes/distilbert/summary.csv | distilbert | cachenotes | 66955010 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> |
| results/unified_cachenotes/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> |
| results/unified_cachenotes/gpt2/summary.csv | gpt2 | cachenotes | 124441344 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> |
| results/unified_cachenotes/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> |
| results/unified_cachenotes/qwen3_5/summary.csv | qwen3_5 | cachenotes | 763217728 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> | 7285760 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> |
| results/unified_cachenotes/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> |
| results/unified_cachenotes/roberta/summary.csv | roberta | cachenotes | 124647170 <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cachenotes | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cachenotes_compressor | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_cachenotes/distilbert/summary.csv | distilbert | full | 44865.75 tok/s <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 2 --> |
| results/unified_cachenotes/distilbert/summary.csv | distilbert | cachenotes | 44353.14 tok/s <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> | 308.81 MB <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> | 0.2403318069239113 <!-- source: results/unified_cachenotes/distilbert/summary.csv line 3 --> |
| results/unified_cachenotes/gpt2/summary.csv | gpt2 | full | 23424.3 tok/s <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 2 --> |
| results/unified_cachenotes/gpt2/summary.csv | gpt2 | cachenotes | 23604.65 tok/s <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> | 637.41 MB <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> | 0.2379362598158943 <!-- source: results/unified_cachenotes/gpt2/summary.csv line 3 --> |
| results/unified_cachenotes/qwen3_5/summary.csv | qwen3_5 | full | 1214.06 tok/s <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> | 1809.78 MB <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 2 --> |
| results/unified_cachenotes/qwen3_5/summary.csv | qwen3_5 | cachenotes | 1046.74 tok/s <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> | 1858.27 MB <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> | 0.0 <!-- source: results/unified_cachenotes/qwen3_5/summary.csv line 3 --> |
| results/unified_cachenotes/roberta/summary.csv | roberta | full | 24050.62 tok/s <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_cachenotes/roberta/summary.csv line 2 --> |
| results/unified_cachenotes/roberta/summary.csv | roberta | cachenotes | 23971.27 tok/s <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> | 527.8 MB <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> | 0.2373566765687054 <!-- source: results/unified_cachenotes/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cachenotes | Elapsed 0.846s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> | 143.6 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 21 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | cachenotes_compressor | Elapsed 0.734s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> | 143.6 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 22 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 81.30% | `results/agnews_unified_cachenotes_compressor/gpt2/summary.csv` |
| F1 Score | 0.8080 | `results/agnews_unified_cachenotes_compressor/gpt2/summary.csv` |
| Compression | 1.000x | `results/agnews_unified_cachenotes_compressor/gpt2/summary.csv` |
| Compressed Layers | 14 | `results/agnews_unified_cachenotes_compressor/gpt2/summary.csv` |
| Fallback Layers | 34 | `results/agnews_unified_cachenotes_compressor/gpt2/summary.csv` |
| Timestamp | 2026-09-18T21:29:35.623759 | `results/agnews_unified_cachenotes_compressor/gpt2/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 90.6, 74.77, 67.55, 89.22, 78.56]
> These numbers must NOT be cited in the final paper.

