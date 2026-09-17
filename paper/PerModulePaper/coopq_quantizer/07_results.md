# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 153.06 ms <!-- source: results/benchmark_summary_v2.csv line 530 --> | 574.9 MB <!-- source: results/benchmark_summary_v2.csv line 530 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 147.68 ms <!-- source: results/benchmark_summary_v2.csv line 531 --> | 574.9 MB <!-- source: results/benchmark_summary_v2.csv line 531 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 146.68 ms <!-- source: results/benchmark_summary_v2.csv line 532 --> | 584.8 MB <!-- source: results/benchmark_summary_v2.csv line 532 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 147.03 ms <!-- source: results/benchmark_summary_v2.csv line 533 --> | 584.8 MB <!-- source: results/benchmark_summary_v2.csv line 533 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 81.05 ms <!-- source: results/benchmark_summary_v2.csv line 534 --> | 1109.1 MB <!-- source: results/benchmark_summary_v2.csv line 534 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 78.05 ms <!-- source: results/benchmark_summary_v2.csv line 535 --> | 1109.1 MB <!-- source: results/benchmark_summary_v2.csv line 535 --> |
| results/all_121_compressors_gpu_matrix.csv | coopq | PASS | Comp Ratio: 1.0x, Time: 0.381s <!-- source: results/all_121_compressors_gpu_matrix.csv line 25 --> | 146.8 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 25 --> |
| results/all_121_compressors_gpu_matrix.csv | coopq_quantizer | PASS | Comp Ratio: 1.0x, Time: 0.359s <!-- source: results/all_121_compressors_gpu_matrix.csv line 26 --> | 146.8 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 26 --> |
