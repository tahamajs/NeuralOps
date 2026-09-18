# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_proxsparse/roberta/summary.csv | proxsparse | PASS_LOW | 50.92% <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> |
| results/unified_proxsparse/distilbert/summary.csv | proxsparse | PASS | 78.33% <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse/gpt2/summary.csv | proxsparse | PASS_LOW | 54.01% <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.87 ms <!-- source: results/benchmark_summary_v2.csv line 662 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 662 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.3 ms <!-- source: results/benchmark_summary_v2.csv line 663 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 663 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 27.45 ms <!-- source: results/benchmark_summary_v2.csv line 664 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 664 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 27.14 ms <!-- source: results/benchmark_summary_v2.csv line 665 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 665 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.7 ms <!-- source: results/benchmark_summary_v2.csv line 666 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 666 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.13 ms <!-- source: results/benchmark_summary_v2.csv line 667 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 667 --> |
| results/all_121_compressors_gpu_matrix.csv | proxsparse | PASS | Comp Ratio: 1.0x, Time: 0.334s <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> | 155.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> |
