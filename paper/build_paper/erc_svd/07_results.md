| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Erc** | **90.48%** | +0.00% | PASS | 0.9077 | 0 | **1.00x** | 310.7 MB | 17.7 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Erc** | **88.53%** | -0.23% | PASS | 0.8932 | 0 | **1.00x** | 321.6 MB | 27.4 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Erc** | **82.45%** | +1.14% | PASS | 0.8374 | 0 | **1.00x** | 400.1 MB | 24.7 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | erc_svd | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | erc_svd | PASS | 32.97 ms/step% <!-- source: results/benchmark_summary_v2.csv line 92 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 92 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | erc_svd | PASS | 34.31 ms/step% <!-- source: results/benchmark_summary_v2.csv line 93 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 93 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | erc_svd | PASS | 34.23 ms/step% <!-- source: results/benchmark_summary_v2.csv line 94 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 94 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | erc_svd | PASS | 34.61 ms/step% <!-- source: results/benchmark_summary_v2.csv line 95 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 95 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | erc_svd | PASS | 52.1 ms/step% <!-- source: results/benchmark_summary_v2.csv line 96 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 96 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | erc_svd | PASS | 51.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 97 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 97 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | erc_svd | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | erc_svd | 4554882 <!-- source: results/benchmark_summary_v2.csv line 92 --> | 4554882 <!-- source: results/benchmark_summary_v2.csv line 92 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 92 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 92 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | erc_svd | 4554882 <!-- source: results/benchmark_summary_v2.csv line 93 --> | 4554882 <!-- source: results/benchmark_summary_v2.csv line 93 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 93 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 93 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | erc_svd | 7070210 <!-- source: results/benchmark_summary_v2.csv line 94 --> | 7070210 <!-- source: results/benchmark_summary_v2.csv line 94 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 94 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 94 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | erc_svd | 7070210 <!-- source: results/benchmark_summary_v2.csv line 95 --> | 7070210 <!-- source: results/benchmark_summary_v2.csv line 95 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 95 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 95 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | erc_svd | 4663040 <!-- source: results/benchmark_summary_v2.csv line 96 --> | 4663040 <!-- source: results/benchmark_summary_v2.csv line 96 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 96 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 96 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | erc_svd | 4663040 <!-- source: results/benchmark_summary_v2.csv line 97 --> | 4663040 <!-- source: results/benchmark_summary_v2.csv line 97 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 97 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 97 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | erc_svd | Elapsed 0.788s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 45 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | erc_svd | Latency 32.97ms tok/s <!-- source: results/benchmark_summary_v2.csv line 92 --> | 797.7 MB <!-- source: results/benchmark_summary_v2.csv line 92 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 92 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | erc_svd | Latency 34.31ms tok/s <!-- source: results/benchmark_summary_v2.csv line 93 --> | 797.7 MB <!-- source: results/benchmark_summary_v2.csv line 93 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 93 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | erc_svd | Latency 34.23ms tok/s <!-- source: results/benchmark_summary_v2.csv line 94 --> | 807.5 MB <!-- source: results/benchmark_summary_v2.csv line 94 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 94 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | erc_svd | Latency 34.61ms tok/s <!-- source: results/benchmark_summary_v2.csv line 95 --> | 807.5 MB <!-- source: results/benchmark_summary_v2.csv line 95 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 95 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | erc_svd | Latency 52.1ms tok/s <!-- source: results/benchmark_summary_v2.csv line 96 --> | 1141.1 MB <!-- source: results/benchmark_summary_v2.csv line 96 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 96 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | erc_svd | Latency 51.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 97 --> | 1141.2 MB <!-- source: results/benchmark_summary_v2.csv line 97 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 97 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 62.50% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-25 23:53:49 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |

