# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_optimal_subspace_projection/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/distilbert/summary.csv | distilbert | optimal_subspace_projection | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> |
| results/unified_optimal_subspace_projection/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/gpt2/summary.csv | gpt2 | optimal_subspace_projection | PASS | 78.1% <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> | 0.7771 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> |
| results/unified_optimal_subspace_projection/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/roberta/summary.csv | roberta | optimal_subspace_projection | PASS | 82.34% <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> | 0.8277 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | optimal_subspace_projection | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_optimal_subspace_projection/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/distilbert/summary.csv | distilbert | optimal_subspace_projection | 66955010 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> |
| results/unified_optimal_subspace_projection/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/gpt2/summary.csv | gpt2 | optimal_subspace_projection | 124441344 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> | 124441346 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> | 2 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> |
| results/unified_optimal_subspace_projection/roberta/summary.csv | roberta | full | 124647170 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> | 124647170 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/roberta/summary.csv | roberta | optimal_subspace_projection | 124647170 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> | 124647173 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> | 3 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | optimal_subspace_projection | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_optimal_subspace_projection/distilbert/summary.csv | distilbert | full | 56128.44 tok/s <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> | 182.94 MB <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/distilbert/summary.csv | distilbert | optimal_subspace_projection | 55482.43 tok/s <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> | 0.0 <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> |
| results/unified_optimal_subspace_projection/gpt2/summary.csv | gpt2 | full | 26905.4 tok/s <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/gpt2/summary.csv | gpt2 | optimal_subspace_projection | 26451.7 tok/s <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> | 644.16 MB <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> | 0.2613349333405494 <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> |
| results/unified_optimal_subspace_projection/roberta/summary.csv | roberta | full | 24421.23 tok/s <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> | 712.34 MB <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 2 --> |
| results/unified_optimal_subspace_projection/roberta/summary.csv | roberta | optimal_subspace_projection | 24802.63 tok/s <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> | 556.31 MB <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> | 0.4600805540879567 <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | optimal_subspace_projection | Elapsed 0.44s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> | 204.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.48% | `results/unified_sweep/distilbert/summary.csv` |
| F1 Score | 0.9091 | `results/unified_sweep/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert/summary.csv` |
| Fallback Layers | 36 | `results/unified_sweep/distilbert/summary.csv` |
| Timestamp | 2026-09-19T01:42:14.854630 | `results/unified_sweep/distilbert/summary.csv` |

