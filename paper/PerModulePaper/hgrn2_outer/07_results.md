# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_hgrn2_outer/roberta/summary.csv | hgrn2_outer | PASS_LOW | 50.34% <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> | 685.64 MB <!-- source: results/unified_hgrn2_outer/roberta/summary.csv line 3 --> |
| results/unified_hgrn2_outer/distilbert/summary.csv | hgrn2_outer | PASS | 68.46% <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> | 329.98 MB <!-- source: results/unified_hgrn2_outer/distilbert/summary.csv line 3 --> |
| results/unified_hgrn2_outer/gpt2/summary.csv | hgrn2_outer | PASS_LOW | 48.62% <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> | 569.81 MB <!-- source: results/unified_hgrn2_outer/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 8683.27 ms <!-- source: results/benchmark_summary_v2.csv line 272 --> | 4895.3 MB <!-- source: results/benchmark_summary_v2.csv line 272 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 8748.24 ms <!-- source: results/benchmark_summary_v2.csv line 273 --> | 4895.3 MB <!-- source: results/benchmark_summary_v2.csv line 273 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 8833.76 ms <!-- source: results/benchmark_summary_v2.csv line 274 --> | 4905.2 MB <!-- source: results/benchmark_summary_v2.csv line 274 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 8761.72 ms <!-- source: results/benchmark_summary_v2.csv line 275 --> | 4905.2 MB <!-- source: results/benchmark_summary_v2.csv line 275 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 8791.15 ms <!-- source: results/benchmark_summary_v2.csv line 276 --> | 4970.8 MB <!-- source: results/benchmark_summary_v2.csv line 276 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 8773.61 ms <!-- source: results/benchmark_summary_v2.csv line 277 --> | 4970.8 MB <!-- source: results/benchmark_summary_v2.csv line 277 --> |
