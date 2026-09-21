# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/unified_delta_kv/distilbert/summary.csv | distilbert | full | PASS | 90.71% <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> | 0.9099 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> |
| results/unified_delta_kv/distilbert/summary.csv | distilbert | delta_kv | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> | 0.9099 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> |
| results/unified_delta_kv/gpt2/summary.csv | gpt2 | full | PASS | 79.36% <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> | 0.7978 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> |
| results/unified_delta_kv/gpt2/summary.csv | gpt2 | delta_kv | PASS_NO_COMPRESSION | 75.92% <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> | 0.7185 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> |
| results/unified_delta_kv/roberta/summary.csv | roberta | full | PASS | 78.56% <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> | 0.8124 <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> |
| results/unified_delta_kv/roberta/summary.csv | roberta | delta_kv | PASS_NO_COMPRESSION | 85.09% <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> | 0.8474 <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | delta_kv | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/unified_delta_kv/distilbert/summary.csv | distilbert | full | 66955010 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> | 66955010 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> | 0 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> | 1.0x <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> |
| results/unified_delta_kv/distilbert/summary.csv | distilbert | delta_kv | 66955010 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> | 66955010 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> | 0 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> | 1.0x <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> |
| results/unified_delta_kv/gpt2/summary.csv | gpt2 | full | 124441344 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> | 124441344 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> | 0 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> | 1.0x <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> |
| results/unified_delta_kv/gpt2/summary.csv | gpt2 | delta_kv | 124441344 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> | 124441344 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> | 0 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> | 1.0x <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> |
| results/unified_delta_kv/roberta/summary.csv | roberta | full | 127325954 <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> | 127325954 <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> | 0 <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> | 1.0x <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> |
| results/unified_delta_kv/roberta/summary.csv | roberta | delta_kv | 127325954 <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> | 127325954 <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> | 2678784 <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> | 1.0x <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | delta_kv | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/unified_delta_kv/distilbert/summary.csv | distilbert | full | 51543.75 tok/s <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> | 389.51 MB <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 2 --> |
| results/unified_delta_kv/distilbert/summary.csv | distilbert | delta_kv | 53326.9 tok/s <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> |
| results/unified_delta_kv/gpt2/summary.csv | gpt2 | full | 26914.89 tok/s <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> | 793.19 MB <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 2 --> |
| results/unified_delta_kv/gpt2/summary.csv | gpt2 | delta_kv | 26456.11 tok/s <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> | 627.91 MB <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> |
| results/unified_delta_kv/roberta/summary.csv | roberta | full | 31018.47 tok/s <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> | 299.92 MB <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> | 0.0000 <!-- source: results/unified_delta_kv/roberta/summary.csv line 2 --> |
| results/unified_delta_kv/roberta/summary.csv | roberta | delta_kv | 12680.67 tok/s <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> | 333.32 MB <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> | 0.0000 <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | delta_kv | Elapsed 0.801s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> | 163.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 95.00% | `results/pareto_table.csv` |
| F1 Score | 0.9464 | `results/pareto_table.csv` |
| Compression | 1.000x | `results/pareto_table.csv` |
| Compressed Layers | 0 | `results/pareto_table.csv` |
| Fallback Layers | 0 | `results/pareto_table.csv` |
| Timestamp | 2026-09-19T16:17:55.513226 | `results/pareto_table.csv` |

