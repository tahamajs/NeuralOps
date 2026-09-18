# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_swift_svd/roberta/summary.csv | swift_svd | PASS_LOW | 50.92% <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> | 629.7 MB <!-- source: results/unified_swift_svd/roberta/summary.csv line 3 --> |
| results/unified_swift_svd/distilbert/summary.csv | swift_svd | PASS | 90.71% <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> | 335.49 MB <!-- source: results/unified_swift_svd/distilbert/summary.csv line 3 --> |
| results/unified_swift_svd/gpt2/summary.csv | swift_svd | PASS_LOW | 50.92% <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> | 755.65 MB <!-- source: results/unified_swift_svd/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 35.1 ms <!-- source: results/benchmark_summary_v2.csv line 116 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 116 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 36.66 ms <!-- source: results/benchmark_summary_v2.csv line 117 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 117 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 36.72 ms <!-- source: results/benchmark_summary_v2.csv line 118 --> | 727.8 MB <!-- source: results/benchmark_summary_v2.csv line 118 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 37.33 ms <!-- source: results/benchmark_summary_v2.csv line 119 --> | 727.8 MB <!-- source: results/benchmark_summary_v2.csv line 119 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 50.19 ms <!-- source: results/benchmark_summary_v2.csv line 120 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 120 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 49.91 ms <!-- source: results/benchmark_summary_v2.csv line 121 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 121 --> |
| results/all_121_compressors_gpu_matrix.csv | swift_svd | PASS | Comp Ratio: 1.0x, Time: 0.883s <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> | 176.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 117 --> |
