# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_armor/roberta/summary.csv | armor_pruner | PASS_NO_COMPRESSION | 79.7% <!-- source: results/unified_armor/roberta/summary.csv line 3 --> | 520.68 MB <!-- source: results/unified_armor/roberta/summary.csv line 3 --> |
| results/unified_armor/distilbert/summary.csv | armor_pruner | PASS_NO_COMPRESSION | 90.02% <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_armor/distilbert/summary.csv line 3 --> |
| results/unified_armor/gpt2/summary.csv | armor_pruner | PASS_NO_COMPRESSION | 49.31% <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> | 627.91 MB <!-- source: results/unified_armor/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 48.09 ms <!-- source: results/benchmark_summary_v2.csv line 608 --> | 843.6 MB <!-- source: results/benchmark_summary_v2.csv line 608 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 43.59 ms <!-- source: results/benchmark_summary_v2.csv line 609 --> | 843.6 MB <!-- source: results/benchmark_summary_v2.csv line 609 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 43.81 ms <!-- source: results/benchmark_summary_v2.csv line 610 --> | 853.5 MB <!-- source: results/benchmark_summary_v2.csv line 610 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 51.33 ms <!-- source: results/benchmark_summary_v2.csv line 611 --> | 853.5 MB <!-- source: results/benchmark_summary_v2.csv line 611 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 64.62 ms <!-- source: results/benchmark_summary_v2.csv line 612 --> | 1251.2 MB <!-- source: results/benchmark_summary_v2.csv line 612 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 64.39 ms <!-- source: results/benchmark_summary_v2.csv line 613 --> | 1251.2 MB <!-- source: results/benchmark_summary_v2.csv line 613 --> |
