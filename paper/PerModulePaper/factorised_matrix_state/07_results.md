# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_factorised_matrix_state/roberta/summary.csv | factorised_matrix_state | PASS_NO_COMPRESSION | 52.87% <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> | 672.35 MB <!-- source: results/unified_factorised_matrix_state/roberta/summary.csv line 3 --> |
| results/unified_factorised_matrix_state/distilbert/summary.csv | factorised_matrix_state | PASS_NO_COMPRESSION | 76.49% <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> | 339.99 MB <!-- source: results/unified_factorised_matrix_state/distilbert/summary.csv line 3 --> |
| results/unified_factorised_matrix_state/gpt2/summary.csv | factorised_matrix_state | PASS_NO_COMPRESSION | 57.8% <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> | 574.03 MB <!-- source: results/unified_factorised_matrix_state/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 6681.92 ms <!-- source: results/benchmark_summary_v2.csv line 50 --> | 3811.3 MB <!-- source: results/benchmark_summary_v2.csv line 50 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 6655.93 ms <!-- source: results/benchmark_summary_v2.csv line 51 --> | 3811.3 MB <!-- source: results/benchmark_summary_v2.csv line 51 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 6668.92 ms <!-- source: results/benchmark_summary_v2.csv line 52 --> | 3821.2 MB <!-- source: results/benchmark_summary_v2.csv line 52 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 6678.48 ms <!-- source: results/benchmark_summary_v2.csv line 53 --> | 3821.2 MB <!-- source: results/benchmark_summary_v2.csv line 53 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 6675.09 ms <!-- source: results/benchmark_summary_v2.csv line 54 --> | 3894.7 MB <!-- source: results/benchmark_summary_v2.csv line 54 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 6917.22 ms <!-- source: results/benchmark_summary_v2.csv line 55 --> | 3894.7 MB <!-- source: results/benchmark_summary_v2.csv line 55 --> |
