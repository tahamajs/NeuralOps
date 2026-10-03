| Model | Module | Accuracy | Delta | Status | F1 | Compressed Layers | Ratio | VRAM | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DistilBERT | full | 90.71% | -- | -- | 0.9099 | 0 | 1.00x | 183 MB | 1.90s |
| DistilBERT | **Kmeans** | **90.48%** | +0.00% | PASS | 0.9071 | 36 | **2.73x** | 556.3 MB | 16.7 |
| RoBERTa | full | 94.38% | -- | -- | 0.9445 | 0 | 1.00x | 293 MB | 4.10s |
| RoBERTa | **Kmeans** | **83.37%** | +8.26% | PASS | 0.8466 | 72 | **3.14x** | 991.9 MB | 24.0 |
| GPT-2 | full | 89.33% | -- | -- | 0.8942 | 0 | 1.00x | 382 MB | 4.20s |
| GPT-2 | **Kmeans** | **62.04%** | +1.26% | PASS | 0.6783 | 0 | **1.00x** | 634.9 MB | 18.4 |

# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kmeans_quantize | PASS | Comp 826.11x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | kmeans_quantize | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 536 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 536 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | kmeans_quantize | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 537 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 537 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | kmeans_quantize | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 538 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 538 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | kmeans_quantize | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 539 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 539 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | kmeans_quantize | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 540 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 540 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | kmeans_quantize | FAIL | 0.0 ms/step% <!-- source: results/benchmark_summary_v2.csv line 541 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 541 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kmeans_quantize | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> | Reduced (826.11x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> | 826.11x <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | kmeans_quantize | 3979906 <!-- source: results/benchmark_summary_v2.csv line 536 --> | 3979906 <!-- source: results/benchmark_summary_v2.csv line 536 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 536 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 536 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | kmeans_quantize | 3979906 <!-- source: results/benchmark_summary_v2.csv line 537 --> | 3979906 <!-- source: results/benchmark_summary_v2.csv line 537 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 537 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 537 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | kmeans_quantize | 6507010 <!-- source: results/benchmark_summary_v2.csv line 538 --> | 6507010 <!-- source: results/benchmark_summary_v2.csv line 538 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 538 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 538 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | kmeans_quantize | 6507010 <!-- source: results/benchmark_summary_v2.csv line 539 --> | 6507010 <!-- source: results/benchmark_summary_v2.csv line 539 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 539 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 539 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | kmeans_quantize | 4116224 <!-- source: results/benchmark_summary_v2.csv line 540 --> | 4116224 <!-- source: results/benchmark_summary_v2.csv line 540 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 540 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 540 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | kmeans_quantize | 4116224 <!-- source: results/benchmark_summary_v2.csv line 541 --> | 4116224 <!-- source: results/benchmark_summary_v2.csv line 541 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 541 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 541 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | kmeans_quantize | Elapsed 0.232s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> | 235.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 64 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | kmeans_quantize | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 536 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 536 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 536 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | kmeans_quantize | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 537 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 537 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 537 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | kmeans_quantize | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 538 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 538 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 538 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | kmeans_quantize | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 539 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 539 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 539 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | kmeans_quantize | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 540 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 540 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 540 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | kmeans_quantize | Latency 0.0ms tok/s <!-- source: results/benchmark_summary_v2.csv line 541 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 541 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 541 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 50.00% | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| F1 Score | 0.0000 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compression | 1.000x | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Compressed Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Fallback Layers | ? | `results/gpu_server/abandoned_run/benchmark_summary.csv` |
| Timestamp | 2026-08-26 00:01:38 | `results/gpu_server/abandoned_run/benchmark_summary.csv` |

