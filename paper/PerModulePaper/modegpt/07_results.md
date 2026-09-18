# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_modegpt/roberta/summary.csv | modegpt | PASS | 72.02% <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_modegpt/roberta/summary.csv line 3 --> |
| results/unified_modegpt/distilbert/summary.csv | modegpt | PASS | 90.48% <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_modegpt/distilbert/summary.csv line 3 --> |
| results/unified_modegpt/gpt2/summary.csv | modegpt | PASS | 60.55% <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> | 639.66 MB <!-- source: results/unified_modegpt/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 21.71 ms <!-- source: results/benchmark_summary_v2.csv line 548 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 548 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 21.63 ms <!-- source: results/benchmark_summary_v2.csv line 549 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 549 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 22.72 ms <!-- source: results/benchmark_summary_v2.csv line 550 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 550 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 23.54 ms <!-- source: results/benchmark_summary_v2.csv line 551 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 551 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.07 ms <!-- source: results/benchmark_summary_v2.csv line 552 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 552 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 51.48 ms <!-- source: results/benchmark_summary_v2.csv line 553 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 553 --> |
| results/all_121_compressors_gpu_matrix.csv | modegpt | PASS | Comp Ratio: 1.0x, Time: 0.901s <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> | 128.5 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 80 --> |
