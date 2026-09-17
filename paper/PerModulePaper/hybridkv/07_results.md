# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_hybridkv/roberta/summary.csv | hybridkv | PASS | 77.18% <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> | 361.56 MB <!-- source: results/unified_hybridkv/roberta/summary.csv line 3 --> |
| results/unified_hybridkv/distilbert/summary.csv | hybridkv | PASS | 90.94% <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> | 337.6 MB <!-- source: results/unified_hybridkv/distilbert/summary.csv line 3 --> |
| results/unified_hybridkv/gpt2/summary.csv | hybridkv | PASS | 62.04% <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> | 634.93 MB <!-- source: results/unified_hybridkv/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.71 ms <!-- source: results/benchmark_summary_v2.csv line 566 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 566 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.63 ms <!-- source: results/benchmark_summary_v2.csv line 567 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 567 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.96 ms <!-- source: results/benchmark_summary_v2.csv line 568 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 568 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 27.47 ms <!-- source: results/benchmark_summary_v2.csv line 569 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 569 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.57 ms <!-- source: results/benchmark_summary_v2.csv line 570 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 570 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.17 ms <!-- source: results/benchmark_summary_v2.csv line 571 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 571 --> |
| results/all_121_compressors_gpu_matrix.csv | hybridkv | PASS | Comp Ratio: 1.0x, Time: 0.17s <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 61 --> |
