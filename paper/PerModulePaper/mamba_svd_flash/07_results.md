# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_mamba_svd_flash/roberta/summary.csv | mamba_svd_flash | FAIL | 0.0% <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> | 298.71 MB <!-- source: results/unified_mamba_svd_flash/roberta/summary.csv line 3 --> |
| results/unified_mamba_svd_flash/distilbert/summary.csv | mamba_svd_flash | FAIL | 0.0% <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> | 182.94 MB <!-- source: results/unified_mamba_svd_flash/distilbert/summary.csv line 3 --> |
| results/unified_mamba_svd_flash/gpt2/summary.csv | mamba_svd_flash | FAIL | 0.0% <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_mamba_svd_flash/gpt2/summary.csv line 3 --> |
| results/master_benchmark_summary.csv | mamba_svd_flash | c4 | "O(N) FLOPs% <!-- source: results/master_benchmark_summary.csv line 2 --> | 72.992 MB <!-- source: results/master_benchmark_summary.csv line 2 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | TIMEOUT | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 56 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 56 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | TIMEOUT | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 57 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 57 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | TIMEOUT | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 58 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 58 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | TIMEOUT | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 59 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 59 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | TIMEOUT | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 60 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 60 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | TIMEOUT | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 61 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 61 --> |
