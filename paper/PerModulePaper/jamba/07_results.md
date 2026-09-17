# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_jamba/roberta/summary.csv | jamba | PASS_LOW | 50.92% <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> | 1321.02 MB <!-- source: results/unified_jamba/roberta/summary.csv line 3 --> |
| results/unified_jamba/distilbert/summary.csv | jamba | PASS | 87.16% <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> | 716.63 MB <!-- source: results/unified_jamba/distilbert/summary.csv line 3 --> |
| results/unified_jamba/gpt2/summary.csv | jamba | PASS_LOW | 52.29% <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> | 1365.61 MB <!-- source: results/unified_jamba/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 3594.17 ms <!-- source: results/benchmark_summary_v2.csv line 356 --> | 3848.8 MB <!-- source: results/benchmark_summary_v2.csv line 356 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 3516.83 ms <!-- source: results/benchmark_summary_v2.csv line 357 --> | 3848.8 MB <!-- source: results/benchmark_summary_v2.csv line 357 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 3526.38 ms <!-- source: results/benchmark_summary_v2.csv line 358 --> | 3858.7 MB <!-- source: results/benchmark_summary_v2.csv line 358 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 3537.3 ms <!-- source: results/benchmark_summary_v2.csv line 359 --> | 3858.7 MB <!-- source: results/benchmark_summary_v2.csv line 359 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 3598.6 ms <!-- source: results/benchmark_summary_v2.csv line 360 --> | 3833.4 MB <!-- source: results/benchmark_summary_v2.csv line 360 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 3578.61 ms <!-- source: results/benchmark_summary_v2.csv line 361 --> | 3833.4 MB <!-- source: results/benchmark_summary_v2.csv line 361 --> |
