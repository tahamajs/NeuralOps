# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_astro/roberta/summary.csv | astro | PASS | 74.77% <!-- source: results/unified_astro/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_astro/roberta/summary.csv line 3 --> |
| results/unified_astro/distilbert/summary.csv | astro | PASS | 90.71% <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_astro/distilbert/summary.csv line 3 --> |
| results/unified_astro/qwen3_5/summary.csv | astro | PASS | 87.5% <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> | 1878.83 MB <!-- source: results/unified_astro/qwen3_5/summary.csv line 3 --> |
| results/unified_astro/gpt2/summary.csv | astro | PASS | 77.87% <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> | 627.28 MB <!-- source: results/unified_astro/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 43.45 ms <!-- source: results/benchmark_summary_v2.csv line 518 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 518 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.25 ms <!-- source: results/benchmark_summary_v2.csv line 519 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 519 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.74 ms <!-- source: results/benchmark_summary_v2.csv line 520 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 520 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 27.31 ms <!-- source: results/benchmark_summary_v2.csv line 521 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 521 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 53.04 ms <!-- source: results/benchmark_summary_v2.csv line 522 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 522 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.4 ms <!-- source: results/benchmark_summary_v2.csv line 523 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 523 --> |
