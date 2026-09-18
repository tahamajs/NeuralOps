# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_swift_svd_act/roberta/summary.csv | swift_svd_act | PASS | 72.02% <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_swift_svd_act/roberta/summary.csv line 3 --> |
| results/unified_swift_svd_act/distilbert/summary.csv | swift_svd_act | PASS | 90.48% <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_swift_svd_act/distilbert/summary.csv line 3 --> |
| results/unified_swift_svd_act/gpt2/summary.csv | swift_svd_act | PASS | 60.09% <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> | 637.29 MB <!-- source: results/unified_swift_svd_act/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 29.96 ms <!-- source: results/benchmark_summary_v2.csv line 122 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 122 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 30.8 ms <!-- source: results/benchmark_summary_v2.csv line 123 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 123 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 30.94 ms <!-- source: results/benchmark_summary_v2.csv line 124 --> | 727.8 MB <!-- source: results/benchmark_summary_v2.csv line 124 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 30.61 ms <!-- source: results/benchmark_summary_v2.csv line 125 --> | 727.8 MB <!-- source: results/benchmark_summary_v2.csv line 125 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 48.9 ms <!-- source: results/benchmark_summary_v2.csv line 126 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 126 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 49.72 ms <!-- source: results/benchmark_summary_v2.csv line 127 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 127 --> |
| results/all_121_compressors_gpu_matrix.csv | swift_svd_act | PASS | Comp Ratio: 30.96x, Time: 0.782s <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 118 --> |
