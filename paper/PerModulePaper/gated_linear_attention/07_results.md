# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_gated_linear_attention/roberta/summary.csv | gated_linear_attention | PASS_LOW | 50.92% <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> | 714.69 MB <!-- source: results/unified_gated_linear_attention/roberta/summary.csv line 3 --> |
| results/unified_gated_linear_attention/distilbert/summary.csv | gated_linear_attention | PASS_LOW | 50.92% <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> | 368.25 MB <!-- source: results/unified_gated_linear_attention/distilbert/summary.csv line 3 --> |
| results/unified_gated_linear_attention/gpt2/summary.csv | gated_linear_attention | PASS_LOW | 50.92% <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> | 607.03 MB <!-- source: results/unified_gated_linear_attention/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 7329.39 ms <!-- source: results/benchmark_summary_v2.csv line 500 --> | 1785.9 MB <!-- source: results/benchmark_summary_v2.csv line 500 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 7170.47 ms <!-- source: results/benchmark_summary_v2.csv line 501 --> | 1785.9 MB <!-- source: results/benchmark_summary_v2.csv line 501 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 7234.59 ms <!-- source: results/benchmark_summary_v2.csv line 502 --> | 1795.8 MB <!-- source: results/benchmark_summary_v2.csv line 502 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 7191.1 ms <!-- source: results/benchmark_summary_v2.csv line 503 --> | 1795.8 MB <!-- source: results/benchmark_summary_v2.csv line 503 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 7286.69 ms <!-- source: results/benchmark_summary_v2.csv line 504 --> | 1853.1 MB <!-- source: results/benchmark_summary_v2.csv line 504 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 7147.66 ms <!-- source: results/benchmark_summary_v2.csv line 505 --> | 1853.1 MB <!-- source: results/benchmark_summary_v2.csv line 505 --> |
