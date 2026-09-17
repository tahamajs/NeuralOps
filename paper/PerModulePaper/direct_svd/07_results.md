# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_direct_svd/roberta/summary.csv | direct_svd | PASS | 71.1% <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> | 567.52 MB <!-- source: results/unified_direct_svd/roberta/summary.csv line 3 --> |
| results/unified_direct_svd/distilbert/summary.csv | direct_svd | PASS | 88.99% <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> | 355.29 MB <!-- source: results/unified_direct_svd/distilbert/summary.csv line 3 --> |
| results/unified_direct_svd/gpt2/summary.csv | direct_svd | PASS | 68.46% <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> | 646.14 MB <!-- source: results/unified_direct_svd/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 34.82 ms <!-- source: results/benchmark_summary_v2.csv line 80 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 80 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 34.71 ms <!-- source: results/benchmark_summary_v2.csv line 81 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 81 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 36.24 ms <!-- source: results/benchmark_summary_v2.csv line 82 --> | 727.9 MB <!-- source: results/benchmark_summary_v2.csv line 82 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 35.76 ms <!-- source: results/benchmark_summary_v2.csv line 83 --> | 727.9 MB <!-- source: results/benchmark_summary_v2.csv line 83 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 50.16 ms <!-- source: results/benchmark_summary_v2.csv line 84 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 84 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 49.88 ms <!-- source: results/benchmark_summary_v2.csv line 85 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 85 --> |
| results/all_121_compressors_gpu_matrix.csv | direct_svd | PASS | Comp Ratio: 30.96x, Time: 0.776s <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 32 --> |
