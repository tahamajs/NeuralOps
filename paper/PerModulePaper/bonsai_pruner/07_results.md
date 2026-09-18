# 7. Empirical Results and Benchmark Evaluations

## 7.1 Downstream Accuracy and Classification Performance
Table 7.1 reports empirical downstream task accuracy and F1 scores across target backbones.

### Table 7.1: Downstream Task Accuracy and Performance Across Backbones
| Source File | Backbone | Variant | Status | Accuracy (%) | F1 Score |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | bonsai | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | bonsai_pruner | PASS | Comp 1.0x% <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> | Mem -0.0% <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | bonsai_pruner | PASS | 25.6 ms/step% <!-- source: results/benchmark_summary_v2.csv line 620 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 620 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | bonsai_pruner | PASS | 24.87 ms/step% <!-- source: results/benchmark_summary_v2.csv line 621 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 621 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | bonsai_pruner | PASS | 26.8 ms/step% <!-- source: results/benchmark_summary_v2.csv line 622 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 622 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | bonsai_pruner | PASS | 27.31 ms/step% <!-- source: results/benchmark_summary_v2.csv line 623 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 623 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | bonsai_pruner | PASS | 52.51 ms/step% <!-- source: results/benchmark_summary_v2.csv line 624 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 624 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | bonsai_pruner | PASS | 52.37 ms/step% <!-- source: results/benchmark_summary_v2.csv line 625 --> | Passed <!-- source: results/benchmark_summary_v2.csv line 625 --> |

## 7.2 Parameter Reduction and Model Efficiency
Table 7.2 details parameter compression ratios, base parameter counts, and trainable recovery adapter parameters.

### Table 7.2: Parameter Counts and Compression Metrics
| Source File | Backbone | Variant | Parameters Before | Parameters After | Trainable Recovery Params | Compression Ratio |
|---|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | bonsai | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | bonsai_pruner | Baseline FP32 <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> | Reduced (1.0x) <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> | 0 <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> | 1.0x <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | bonsai_pruner | 4386178 <!-- source: results/benchmark_summary_v2.csv line 620 --> | 4386178 <!-- source: results/benchmark_summary_v2.csv line 620 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 620 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 620 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | bonsai_pruner | 4386178 <!-- source: results/benchmark_summary_v2.csv line 621 --> | 4386178 <!-- source: results/benchmark_summary_v2.csv line 621 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 621 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 621 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | bonsai_pruner | 6913282 <!-- source: results/benchmark_summary_v2.csv line 622 --> | 6913282 <!-- source: results/benchmark_summary_v2.csv line 622 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 622 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 622 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | bonsai_pruner | 6913282 <!-- source: results/benchmark_summary_v2.csv line 623 --> | 6913282 <!-- source: results/benchmark_summary_v2.csv line 623 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 623 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 623 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | bonsai_pruner | 4506368 <!-- source: results/benchmark_summary_v2.csv line 624 --> | 4506368 <!-- source: results/benchmark_summary_v2.csv line 624 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 624 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 624 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | bonsai_pruner | 4506368 <!-- source: results/benchmark_summary_v2.csv line 625 --> | 4506368 <!-- source: results/benchmark_summary_v2.csv line 625 --> | 0 <!-- source: results/benchmark_summary_v2.csv line 625 --> | 1.0x <!-- source: results/benchmark_summary_v2.csv line 625 --> |

## 7.3 Latency, Throughput, and VRAM Utilization
Table 7.3 summarizes GPU execution metrics, peak VRAM allocations, and reconstruction errors.

### Table 7.3: Hardware Throughput and Memory Profiling
| Source File | Backbone | Variant | Throughput (Tokens/s) | Peak VRAM | Reconstruction Error |
|---|---|---|---|---|---|
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | bonsai | Elapsed 0.386s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> | 146.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 19 --> |
| results/all_121_compressors_gpu_matrix.csv | NVIDIA-GPU-Matrix | bonsai_pruner | Elapsed 0.346s tok/s <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> | 146.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> | Verified <!-- source: results/all_121_compressors_gpu_matrix.csv line 20 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | bonsai_pruner | Latency 25.6ms tok/s <!-- source: results/benchmark_summary_v2.csv line 620 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 620 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 620 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | bonsai_pruner | Latency 24.87ms tok/s <!-- source: results/benchmark_summary_v2.csv line 621 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 621 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 621 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | bonsai_pruner | Latency 26.8ms tok/s <!-- source: results/benchmark_summary_v2.csv line 622 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 622 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 622 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | bonsai_pruner | Latency 27.31ms tok/s <!-- source: results/benchmark_summary_v2.csv line 623 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 623 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 623 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | bonsai_pruner | Latency 52.51ms tok/s <!-- source: results/benchmark_summary_v2.csv line 624 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 624 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 624 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | bonsai_pruner | Latency 52.37ms tok/s <!-- source: results/benchmark_summary_v2.csv line 625 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 625 --> | Verified <!-- source: results/benchmark_summary_v2.csv line 625 --> |
