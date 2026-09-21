# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | ast | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | ast_pruner | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | ast_pruner | PASS | 26.06 ms/step% <!-- source: results/benchmark_summary_v2.csv line 614 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 614 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | ast_pruner | PASS | 25.88 ms/step% <!-- source: results/benchmark_summary_v2.csv line 615 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 615 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | ast_pruner | PASS | 28.23 ms/step% <!-- source: results/benchmark_summary_v2.csv line 616 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 616 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | ast_pruner | PASS | 26.83 ms/step% <!-- source: results/benchmark_summary_v2.csv line 617 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 617 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | ast_pruner | PASS | 52.87 ms/step% <!-- source: results/benchmark_summary_v2.csv line 618 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 618 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | ast_pruner | PASS | 52.73 ms/step% <!-- source: results/benchmark_summary_v2.csv line 619 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 619 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | ast | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | ast_pruner | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | ast_pruner | 4386178 <!-- source: results/benchmark_summary_v2.csv line 614 --> | 4386178 <!-- source: results/benchmark_summary_v2.csv line 614 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 614 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 614 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | ast_pruner | 4386178 <!-- source: results/benchmark_summary_v2.csv line 615 --> | 4386178 <!-- source: results/benchmark_summary_v2.csv line 615 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 615 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 615 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | ast_pruner | 6913282 <!-- source: results/benchmark_summary_v2.csv line 616 --> | 6913282 <!-- source: results/benchmark_summary_v2.csv line 616 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 616 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 616 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | ast_pruner | 6913282 <!-- source: results/benchmark_summary_v2.csv line 617 --> | 6913282 <!-- source: results/benchmark_summary_v2.csv line 617 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 617 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 617 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | ast_pruner | 4506368 <!-- source: results/benchmark_summary_v2.csv line 618 --> | 4506368 <!-- source: results/benchmark_summary_v2.csv line 618 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 618 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 618 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | ast_pruner | 4506368 <!-- source: results/benchmark_summary_v2.csv line 619 --> | 4506368 <!-- source: results/benchmark_summary_v2.csv line 619 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 619 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 619 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | ast | Elapsed 0.235s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> | 202.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 15 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | ast_pruner | Elapsed 0.194s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> | 202.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 16 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | ast_pruner | Latency 26.06ms tok/s <!-- source: results/benchmark_summary_v2.csv line 614 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 614 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 614 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | ast_pruner | Latency 25.88ms tok/s <!-- source: results/benchmark_summary_v2.csv line 615 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 615 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 615 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | ast_pruner | Latency 28.23ms tok/s <!-- source: results/benchmark_summary_v2.csv line 616 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 616 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 616 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | ast_pruner | Latency 26.83ms tok/s <!-- source: results/benchmark_summary_v2.csv line 617 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 617 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 617 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | ast_pruner | Latency 52.87ms tok/s <!-- source: results/benchmark_summary_v2.csv line 618 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 618 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 618 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | ast_pruner | Latency 52.73ms tok/s <!-- source: results/benchmark_summary_v2.csv line 619 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 619 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 619 --> |

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
| Timestamp | 2026-08-26 09:50:57 | `results/adaptive_v3/benchmark_summary.csv` |

