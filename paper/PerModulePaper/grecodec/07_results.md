# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_grecodec/roberta/summary.csv | grecodec | PASS | 88.53% <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> | 321.64 MB <!-- source: results/unified_grecodec/roberta/summary.csv line 3 --> |
| results/unified_grecodec/distilbert/summary.csv | grecodec | PASS | 90.37% <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> | 310.69 MB <!-- source: results/unified_grecodec/distilbert/summary.csv line 3 --> |
| results/unified_grecodec/gpt2/summary.csv | grecodec | PASS | 82.57% <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> | 398.74 MB <!-- source: results/unified_grecodec/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 35.34 ms <!-- source: results/benchmark_summary_v2.csv line 140 --> | 581.9 MB <!-- source: results/benchmark_summary_v2.csv line 140 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 32.18 ms <!-- source: results/benchmark_summary_v2.csv line 141 --> | 581.9 MB <!-- source: results/benchmark_summary_v2.csv line 141 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 37.38 ms <!-- source: results/benchmark_summary_v2.csv line 142 --> | 591.8 MB <!-- source: results/benchmark_summary_v2.csv line 142 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 37.77 ms <!-- source: results/benchmark_summary_v2.csv line 143 --> | 591.8 MB <!-- source: results/benchmark_summary_v2.csv line 143 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 49.13 ms <!-- source: results/benchmark_summary_v2.csv line 144 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 144 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 49.54 ms <!-- source: results/benchmark_summary_v2.csv line 145 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 145 --> |
| results/all_121_compressors_gpu_matrix.csv | grecodec | PASS | Comp Ratio: 30.96x, Time: 6.41s <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> | 101.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 56 --> |
