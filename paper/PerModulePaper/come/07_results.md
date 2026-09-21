# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_come/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_come/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_come/distilbert/summary.csv line 2 --> |
| results/unified_come/distilbert/summary.csv | distilbert | come | PASS | 84.29% <!-- source: results/unified_come/distilbert/summary.csv line 3 --> | 0.8355 <!-- source: results/unified_come/distilbert/summary.csv line 3 --> |
| results/unified_come/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_come/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_come/gpt2/summary.csv line 2 --> |
| results/unified_come/gpt2/summary.csv | gpt2 | come | FAIL | 0.0% <!-- source: results/unified_come/gpt2/summary.csv line 3 --> | 0.7978 <!-- source: results/unified_come/gpt2/summary.csv line 3 --> |
| results/unified_come/qwen3_5/summary.csv | qwen3_5 | full | PASS | 67.89% <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> | 0.557 <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> |
| results/unified_come/qwen3_5/summary.csv | qwen3_5 | come | FAIL | 0.0% <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> | 0.557 <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> |
| results/unified_come/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_come/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_come/roberta/summary.csv line 2 --> |
| results/unified_come/roberta/summary.csv | roberta | come | FAIL | 0.0% <!-- source: results/unified_come/roberta/summary.csv line 3 --> | 0.8124 <!-- source: results/unified_come/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | come | PASS | Comp 2.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_come/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_come/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_come/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_come/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_come/distilbert/summary.csv line 2 --> |
| results/unified_come/distilbert/summary.csv | distilbert | come | 66955010 <!-- source: results/unified_come/distilbert/summary.csv line 3 --> | 53958916 <!-- source: results/unified_come/distilbert/summary.csv line 3 --> | 2 <!-- source: results/unified_come/distilbert/summary.csv line 3 --> | 1.2409x <!-- source: results/unified_come/distilbert/summary.csv line 3 --> |
| results/unified_come/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_come/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_come/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_come/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_come/gpt2/summary.csv line 2 --> |
| results/unified_come/gpt2/summary.csv | gpt2 | come | 124441344 <!-- source: results/unified_come/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_come/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_come/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_come/gpt2/summary.csv line 3 --> |
| results/unified_come/qwen3_5/summary.csv | qwen3_5 | full | 763217728 <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> | 763217728 <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> | 0 <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> | 1.0x <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> |
| results/unified_come/qwen3_5/summary.csv | qwen3_5 | come | 763217728 <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> | 763217728 <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> | 0 <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> | 1.0x <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> |
| results/unified_come/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_come/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_come/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_come/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_come/roberta/summary.csv line 2 --> |
| results/unified_come/roberta/summary.csv | roberta | come | 124647170 <!-- source: results/unified_come/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_come/roberta/summary.csv line 3 --> | 0 <!-- source: results/unified_come/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_come/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | come | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> | Reduced (2.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> | 2.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_come/distilbert/summary.csv | distilbert | full | 45077.54 tok/s <!-- source: results/unified_come/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_come/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_come/distilbert/summary.csv line 2 --> |
| results/unified_come/distilbert/summary.csv | distilbert | come | 58463.31 tok/s <!-- source: results/unified_come/distilbert/summary.csv line 3 --> | 259.48 MB <!-- source: results/unified_come/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_come/distilbert/summary.csv line 3 --> |
| results/unified_come/gpt2/summary.csv | gpt2 | full | 25258.69 tok/s <!-- source: results/unified_come/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_come/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_come/gpt2/summary.csv line 2 --> |
| results/unified_come/gpt2/summary.csv | gpt2 | come | 25258.69 tok/s <!-- source: results/unified_come/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_come/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_come/gpt2/summary.csv line 3 --> |
| results/unified_come/qwen3_5/summary.csv | qwen3_5 | full | 1289.77 tok/s <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> | 1796.78 MB <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_come/qwen3_5/summary.csv line 2 --> |
| results/unified_come/qwen3_5/summary.csv | qwen3_5 | come | 1289.77 tok/s <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> | 1796.78 MB <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> |
| results/unified_come/roberta/summary.csv | roberta | full | 25968.34 tok/s <!-- source: results/unified_come/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_come/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_come/roberta/summary.csv line 2 --> |
| results/unified_come/roberta/summary.csv | roberta | come | 25968.34 tok/s <!-- source: results/unified_come/roberta/summary.csv line 3 --> | 712.34 MB <!-- source: results/unified_come/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_come/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | come | Elapsed 0.161s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> | 72.9 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=91.50%, comp=1.241x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 91.50% | `results/pareto_table.csv` |
| F1 Score | 0.9104 | `results/pareto_table.csv` |
| Compression | 1.241x | `results/pareto_table.csv` |
| Compressed Layers | 1 | `results/pareto_table.csv` |
| Fallback Layers | 0 | `results/pareto_table.csv` |
| Timestamp | 2026-09-19T16:12:55.658957 | `results/pareto_table.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [67.89]
> These numbers must NOT be cited in the final paper.

