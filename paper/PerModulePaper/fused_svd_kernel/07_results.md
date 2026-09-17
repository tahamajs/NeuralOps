# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_fused_svd_kernel/roberta/summary.csv | fused_svd_kernel | PASS_LOW | 50.92% <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> | 558.41 MB <!-- source: results/unified_fused_svd_kernel/roberta/summary.csv line 3 --> |
| results/unified_fused_svd_kernel/distilbert/summary.csv | fused_svd_kernel | PASS | 79.93% <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> | 280.27 MB <!-- source: results/unified_fused_svd_kernel/distilbert/summary.csv line 3 --> |
| results/unified_fused_svd_kernel/gpt2/summary.csv | fused_svd_kernel | PASS | 70.07% <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> | 464.46 MB <!-- source: results/unified_fused_svd_kernel/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | FAIL | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 512 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 512 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | FAIL | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 513 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 513 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | FAIL | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 514 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 514 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | FAIL | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 515 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 515 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | FAIL | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 516 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 516 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | FAIL | Step Latency: 0.0 ms <!-- source: results/benchmark_summary_v2.csv line 517 --> | 0.0 MB <!-- source: results/benchmark_summary_v2.csv line 517 --> |
