# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_kvtc/roberta/summary.csv | kvtc | PASS_LOW | 50.92% <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> | 457.87 MB <!-- source: results/unified_kvtc/roberta/summary.csv line 3 --> |
| results/unified_kvtc/distilbert/summary.csv | kvtc | PASS | 90.14% <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> | 217.23 MB <!-- source: results/unified_kvtc/distilbert/summary.csv line 3 --> |
| results/unified_kvtc/gpt2/summary.csv | kvtc | FAIL | 0.0% <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_kvtc/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.3 ms <!-- source: results/benchmark_summary_v2.csv line 560 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 560 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 26.11 ms <!-- source: results/benchmark_summary_v2.csv line 561 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 561 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 27.44 ms <!-- source: results/benchmark_summary_v2.csv line 562 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 562 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 26.2 ms <!-- source: results/benchmark_summary_v2.csv line 563 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 563 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.7 ms <!-- source: results/benchmark_summary_v2.csv line 564 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 564 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.19 ms <!-- source: results/benchmark_summary_v2.csv line 565 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 565 --> |
| results/all_121_compressors_gpu_matrix.csv | kvtc | PASS | Comp Ratio: 1.0x, Time: 0.789s <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> | 147.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 67 --> |
