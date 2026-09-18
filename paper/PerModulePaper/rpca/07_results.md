# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_rpca/roberta/summary.csv | rpca | PASS | 72.02% <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_rpca/roberta/summary.csv line 3 --> |
| results/unified_rpca/distilbert/summary.csv | rpca | PASS | 90.37% <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_rpca/distilbert/summary.csv line 3 --> |
| results/unified_rpca/gpt2/summary.csv | rpca | PASS | 60.21% <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> | 637.67 MB <!-- source: results/unified_rpca/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 35.07 ms <!-- source: results/benchmark_summary_v2.csv line 134 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 134 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 35.19 ms <!-- source: results/benchmark_summary_v2.csv line 135 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 135 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 37.67 ms <!-- source: results/benchmark_summary_v2.csv line 136 --> | 727.8 MB <!-- source: results/benchmark_summary_v2.csv line 136 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 36.35 ms <!-- source: results/benchmark_summary_v2.csv line 137 --> | 727.8 MB <!-- source: results/benchmark_summary_v2.csv line 137 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 49.77 ms <!-- source: results/benchmark_summary_v2.csv line 138 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 138 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 50.41 ms <!-- source: results/benchmark_summary_v2.csv line 139 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 139 --> |
| results/all_121_compressors_gpu_matrix.csv | rpca | PASS | Comp Ratio: 30.96x, Time: 3.946s <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> | 155.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 99 --> |
