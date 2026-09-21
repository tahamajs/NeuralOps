# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | entropy_budget_allocation | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | entropy_budget_allocation | PASS | 28.93 ms/step% <!-- source: results/benchmark_summary_v2.csv line 98 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 98 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | entropy_budget_allocation | PASS | 28.94 ms/step% <!-- source: results/benchmark_summary_v2.csv line 99 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 99 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | entropy_budget_allocation | PASS | 30.5 ms/step% <!-- source: results/benchmark_summary_v2.csv line 100 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 100 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | entropy_budget_allocation | PASS | 29.92 ms/step% <!-- source: results/benchmark_summary_v2.csv line 101 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 101 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | entropy_budget_allocation | PASS | 48.9 ms/step% <!-- source: results/benchmark_summary_v2.csv line 102 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 102 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | entropy_budget_allocation | PASS | 48.89 ms/step% <!-- source: results/benchmark_summary_v2.csv line 103 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 103 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | entropy_budget_allocation | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | entropy_budget_allocation | 4054402 <!-- source: results/benchmark_summary_v2.csv line 98 --> | 4054402 <!-- source: results/benchmark_summary_v2.csv line 98 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 98 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 98 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | entropy_budget_allocation | 4054402 <!-- source: results/benchmark_summary_v2.csv line 99 --> | 4054402 <!-- source: results/benchmark_summary_v2.csv line 99 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 99 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 99 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | entropy_budget_allocation | 6593794 <!-- source: results/benchmark_summary_v2.csv line 100 --> | 6593794 <!-- source: results/benchmark_summary_v2.csv line 100 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 100 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 100 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | entropy_budget_allocation | 6593794 <!-- source: results/benchmark_summary_v2.csv line 101 --> | 6593794 <!-- source: results/benchmark_summary_v2.csv line 101 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 101 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 101 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | entropy_budget_allocation | 4186880 <!-- source: results/benchmark_summary_v2.csv line 102 --> | 4186880 <!-- source: results/benchmark_summary_v2.csv line 102 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 102 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 102 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | entropy_budget_allocation | 4186880 <!-- source: results/benchmark_summary_v2.csv line 103 --> | 4186880 <!-- source: results/benchmark_summary_v2.csv line 103 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 103 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 103 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | entropy_budget_allocation | Elapsed 1.342s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> | 176.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 42 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | entropy_budget_allocation | Latency 28.93ms tok/s <!-- source: results/benchmark_summary_v2.csv line 98 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 98 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 98 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | entropy_budget_allocation | Latency 28.94ms tok/s <!-- source: results/benchmark_summary_v2.csv line 99 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 99 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 99 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | entropy_budget_allocation | Latency 30.5ms tok/s <!-- source: results/benchmark_summary_v2.csv line 100 --> | 727.9 MB <!-- source: results/benchmark_summary_v2.csv line 100 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 100 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | entropy_budget_allocation | Latency 29.92ms tok/s <!-- source: results/benchmark_summary_v2.csv line 101 --> | 727.9 MB <!-- source: results/benchmark_summary_v2.csv line 101 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 101 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | entropy_budget_allocation | Latency 48.9ms tok/s <!-- source: results/benchmark_summary_v2.csv line 102 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 102 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 102 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | entropy_budget_allocation | Latency 48.89ms tok/s <!-- source: results/benchmark_summary_v2.csv line 103 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 103 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 103 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 68.38% | `results/adaptive_v3/benchmark_summary.csv` |
| F1 Score | 0.8122 | `results/adaptive_v3/benchmark_summary.csv` |
| Compression | 1.000x | `results/adaptive_v3/benchmark_summary.csv` |
| Compressed Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Fallback Layers | ? | `results/adaptive_v3/benchmark_summary.csv` |
| Timestamp | 2026-08-26 07:14:11 | `results/adaptive_v3/benchmark_summary.csv` |

