# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dynamic_rank | PASS | Comp 30.96x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | dynamic_rank | PASS | 34.47 ms/step% <!-- source: results/benchmark_summary_v2.csv line 152 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 152 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | dynamic_rank | PASS | 34.7 ms/step% <!-- source: results/benchmark_summary_v2.csv line 153 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 153 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | dynamic_rank | PASS | 34.92 ms/step% <!-- source: results/benchmark_summary_v2.csv line 154 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 154 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | dynamic_rank | PASS | 30.93 ms/step% <!-- source: results/benchmark_summary_v2.csv line 155 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 155 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | dynamic_rank | PASS | 49.47 ms/step% <!-- source: results/benchmark_summary_v2.csv line 156 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 156 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | dynamic_rank | PASS | 50.3 ms/step% <!-- source: results/benchmark_summary_v2.csv line 157 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 157 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dynamic_rank | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> | Reduced (30.96x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> | 30.96x <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | dynamic_rank | 4054402 <!-- source: results/benchmark_summary_v2.csv line 152 --> | 4054402 <!-- source: results/benchmark_summary_v2.csv line 152 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 152 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 152 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | dynamic_rank | 4054402 <!-- source: results/benchmark_summary_v2.csv line 153 --> | 4054402 <!-- source: results/benchmark_summary_v2.csv line 153 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 153 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 153 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | dynamic_rank | 6593794 <!-- source: results/benchmark_summary_v2.csv line 154 --> | 6593794 <!-- source: results/benchmark_summary_v2.csv line 154 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 154 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 154 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | dynamic_rank | 6593794 <!-- source: results/benchmark_summary_v2.csv line 155 --> | 6593794 <!-- source: results/benchmark_summary_v2.csv line 155 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 155 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 155 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | dynamic_rank | 4186880 <!-- source: results/benchmark_summary_v2.csv line 156 --> | 4186880 <!-- source: results/benchmark_summary_v2.csv line 156 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 156 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 156 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | dynamic_rank | 4186880 <!-- source: results/benchmark_summary_v2.csv line 157 --> | 4186880 <!-- source: results/benchmark_summary_v2.csv line 157 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 157 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 157 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | dynamic_rank | Elapsed 1.007s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> | 104.9 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 39 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | dynamic_rank | Latency 34.47ms tok/s <!-- source: results/benchmark_summary_v2.csv line 152 --> | 581.9 MB <!-- source: results/benchmark_summary_v2.csv line 152 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 152 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | dynamic_rank | Latency 34.7ms tok/s <!-- source: results/benchmark_summary_v2.csv line 153 --> | 581.9 MB <!-- source: results/benchmark_summary_v2.csv line 153 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 153 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | dynamic_rank | Latency 34.92ms tok/s <!-- source: results/benchmark_summary_v2.csv line 154 --> | 591.9 MB <!-- source: results/benchmark_summary_v2.csv line 154 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 154 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | dynamic_rank | Latency 30.93ms tok/s <!-- source: results/benchmark_summary_v2.csv line 155 --> | 591.9 MB <!-- source: results/benchmark_summary_v2.csv line 155 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 155 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | dynamic_rank | Latency 49.47ms tok/s <!-- source: results/benchmark_summary_v2.csv line 156 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 156 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 156 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | dynamic_rank | Latency 50.3ms tok/s <!-- source: results/benchmark_summary_v2.csv line 157 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 157 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 157 --> |

---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 94.40% | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| F1 Score | 0.9404 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Compressed Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |
| Timestamp | 2026-09-19T16:18:30.755725 | `results/unified_sweep/distilbert_mnli_full/summary.csv` |

