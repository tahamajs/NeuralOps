# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_swift_svd_act/distilbert/summary.csv | distilbert | full | PASS | 90.48% <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> | 0.9077 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> |
| results/unified_swift_svd_act/distilbert/summary.csv | distilbert | swift_svd_act | PASS | 90.48% <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> | 0.9077 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> |
| results/unified_swift_svd_act/gpt2/summary.csv | gpt2 | full | PASS | 60.78% <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> | 0.6354 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> |
| results/unified_swift_svd_act/gpt2/summary.csv | gpt2 | swift_svd_act | PASS | 60.09% <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> | 0.6463 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> |
| results/unified_swift_svd_act/roberta/summary.csv | roberta | full | PASS | 75.11% <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> | 0.7586 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> |
| results/unified_swift_svd_act/roberta/summary.csv | roberta | swift_svd_act | PASS | 72.02% <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> | 0.7724 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | swift_svd_act | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_swift_svd_act/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> |
| results/unified_swift_svd_act/distilbert/summary.csv | distilbert | swift_svd_act | 66955010 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> | 592130 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> |
| results/unified_swift_svd_act/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> |
| results/unified_swift_svd_act/gpt2/summary.csv | gpt2 | swift_svd_act | 124441344 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> | 124048129 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> | 198913 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> | 1.0032x <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> |
| results/unified_swift_svd_act/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> |
| results/unified_swift_svd_act/roberta/summary.csv | roberta | swift_svd_act | 124647170 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> | 124647170 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> | 592130 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | swift_svd_act | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_swift_svd_act/distilbert/summary.csv | distilbert | full | 135590.4 tok/s <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 2 --> |
| results/unified_swift_svd_act/distilbert/summary.csv | distilbert | swift_svd_act | 146011.97 tok/s <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> |
| results/unified_swift_svd_act/gpt2/summary.csv | gpt2 | full | 67936.84 tok/s <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 2 --> |
| results/unified_swift_svd_act/gpt2/summary.csv | gpt2 | swift_svd_act | 66501.09 tok/s <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> | 637.29 MB <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> | 0.4203306436538696 <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> |
| results/unified_swift_svd_act/roberta/summary.csv | roberta | full | 88009.53 tok/s <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> | 298.71 MB <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 2 --> |
| results/unified_swift_svd_act/roberta/summary.csv | roberta | swift_svd_act | 91839.43 tok/s <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> | 0.0 <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | swift_svd_act | Elapsed 0.782s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `VALID`  
**Detail**: acc=60.09%, comp=1.003x

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 60.09% | `results/unified_swift_svd_act/gpt2/summary.csv` |
| F1 Score | 0.6463 | `results/unified_swift_svd_act/gpt2/summary.csv` |
| Compression | 1.003x | `results/unified_swift_svd_act/gpt2/summary.csv` |
| Compressed Layers | 1 | `results/unified_swift_svd_act/gpt2/summary.csv` |
| Fallback Layers | 47 | `results/unified_swift_svd_act/gpt2/summary.csv` |
| Timestamp | 2026-09-11T09:40:02.248668 | `results/unified_swift_svd_act/gpt2/summary.csv` |

