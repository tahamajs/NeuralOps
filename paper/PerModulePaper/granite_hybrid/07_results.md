# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_granite_hybrid/roberta/summary.csv | granite_hybrid | PASS_LOW | 53.33% <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> | 2501.8 MB <!-- source: results/unified_granite_hybrid/roberta/summary.csv line 3 --> |
| results/unified_granite_hybrid/distilbert/summary.csv | granite_hybrid | PASS | 63.65% <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> | 1604.75 MB <!-- source: results/unified_granite_hybrid/distilbert/summary.csv line 3 --> |
| results/unified_granite_hybrid/gpt2/summary.csv | granite_hybrid | PASS_LOW | 51.72% <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> | 2393.89 MB <!-- source: results/unified_granite_hybrid/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 1704.51 ms <!-- source: results/benchmark_summary_v2.csv line 440 --> | 4166.7 MB <!-- source: results/benchmark_summary_v2.csv line 440 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 1707.27 ms <!-- source: results/benchmark_summary_v2.csv line 441 --> | 4166.7 MB <!-- source: results/benchmark_summary_v2.csv line 441 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 1714.33 ms <!-- source: results/benchmark_summary_v2.csv line 442 --> | 4176.6 MB <!-- source: results/benchmark_summary_v2.csv line 442 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 1709.78 ms <!-- source: results/benchmark_summary_v2.csv line 443 --> | 4176.6 MB <!-- source: results/benchmark_summary_v2.csv line 443 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 1719.55 ms <!-- source: results/benchmark_summary_v2.csv line 444 --> | 4265.1 MB <!-- source: results/benchmark_summary_v2.csv line 444 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 1711.84 ms <!-- source: results/benchmark_summary_v2.csv line 445 --> | 4265.1 MB <!-- source: results/benchmark_summary_v2.csv line 445 --> |
