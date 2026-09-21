# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_rpca/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> |
| results/unified_rpca/distilbert/summary.csv | distilbert | rpca | PASS | 90.37% <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> | 0.9067 <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> |
| results/unified_rpca/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> |
| results/unified_rpca/gpt2/summary.csv | gpt2 | rpca | PASS | 60.21% <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> | 0.6825 <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> |
| results/unified_rpca/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> |
| results/unified_rpca/roberta/summary.csv | roberta | rpca | PASS | 72.02% <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> | 0.7724 <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | rpca | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_rpca/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> |
| results/unified_rpca/distilbert/summary.csv | distilbert | rpca | 66955010 <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> |
| results/unified_rpca/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> |
| results/unified_rpca/gpt2/summary.csv | gpt2 | rpca | 124441344 <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> | 124048129 <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> | 198913 <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> | 1.0032x <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> |
| results/unified_rpca/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> |
| results/unified_rpca/roberta/summary.csv | roberta | rpca | 124647170 <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | rpca | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_rpca/distilbert/summary.csv | distilbert | full | 128841.24 tok/s <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_rpca/distilbert/summary.csv line 2 --> |
| results/unified_rpca/distilbert/summary.csv | distilbert | rpca | 138163.61 tok/s <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> |
| results/unified_rpca/gpt2/summary.csv | gpt2 | full | 63915.32 tok/s <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_rpca/gpt2/summary.csv line 2 --> |
| results/unified_rpca/gpt2/summary.csv | gpt2 | rpca | 65063.68 tok/s <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> | 637.67 MB <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> | 0.4204952716827392 <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> |
| results/unified_rpca/roberta/summary.csv | roberta | full | 94654.06 tok/s <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_rpca/roberta/summary.csv line 2 --> |
| results/unified_rpca/roberta/summary.csv | roberta | rpca | 92321.44 tok/s <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | rpca | Elapsed 3.946s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> | 155.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=60.21%, comp=1.003x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 60.21% | `results/unified_rpca/gpt2/summary.csv` |
| F1 Score | 0.6825 | `results/unified_rpca/gpt2/summary.csv` |
| Compression | 1.003x | `results/unified_rpca/gpt2/summary.csv` |
| Compressed Layers | 1 | `results/unified_rpca/gpt2/summary.csv` |
| Fallback Layers | 0 | `results/unified_rpca/gpt2/summary.csv` |
| Timestamp | 2026-09-11T08:05:41.849179 | `results/unified_rpca/gpt2/summary.csv` |

