# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_delta_kv/roberta/summary.csv | delta_kv | PASS_NO_COMPRESSION | 85.09% <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> | 333.32 MB <!-- source: results/unified_delta_kv/roberta/summary.csv line 3 --> |
| results/unified_delta_kv/distilbert/summary.csv | delta_kv | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_delta_kv/distilbert/summary.csv line 3 --> |
| results/unified_delta_kv/gpt2/summary.csv | delta_kv | PASS_NO_COMPRESSION | 75.92% <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> | 627.91 MB <!-- source: results/unified_delta_kv/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 21.9 ms <!-- source: results/benchmark_summary_v2.csv line 554 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 554 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 22.55 ms <!-- source: results/benchmark_summary_v2.csv line 555 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 555 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.83 ms <!-- source: results/benchmark_summary_v2.csv line 556 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 556 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 27.17 ms <!-- source: results/benchmark_summary_v2.csv line 557 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 557 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.67 ms <!-- source: results/benchmark_summary_v2.csv line 558 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 558 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.33 ms <!-- source: results/benchmark_summary_v2.csv line 559 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 559 --> |
| results/all_121_compressors_gpu_matrix.csv | delta_kv | PASS | Comp Ratio: 1.0x, Time: 0.801s <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> | 163.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 31 --> |
