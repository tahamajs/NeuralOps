# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_cur_moe/roberta/summary.csv | cur_moe | PASS_LOW | 50.92% <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> | 615.57 MB <!-- source: results/unified_cur_moe/roberta/summary.csv line 3 --> |
| results/unified_cur_moe/distilbert/summary.csv | cur_moe | PASS | 74.43% <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> | 253.68 MB <!-- source: results/unified_cur_moe/distilbert/summary.csv line 3 --> |
| results/unified_cur_moe/gpt2/summary.csv | cur_moe | PASS_LOW | 50.92% <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> | 625.48 MB <!-- source: results/unified_cur_moe/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 44.11 ms <!-- source: results/benchmark_summary_v2.csv line 446 --> | 594.0 MB <!-- source: results/benchmark_summary_v2.csv line 446 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 44.49 ms <!-- source: results/benchmark_summary_v2.csv line 447 --> | 594.0 MB <!-- source: results/benchmark_summary_v2.csv line 447 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 45.24 ms <!-- source: results/benchmark_summary_v2.csv line 448 --> | 603.9 MB <!-- source: results/benchmark_summary_v2.csv line 448 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 45.39 ms <!-- source: results/benchmark_summary_v2.csv line 449 --> | 603.9 MB <!-- source: results/benchmark_summary_v2.csv line 449 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.38 ms <!-- source: results/benchmark_summary_v2.csv line 450 --> | 1091.2 MB <!-- source: results/benchmark_summary_v2.csv line 450 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.83 ms <!-- source: results/benchmark_summary_v2.csv line 451 --> | 1091.2 MB <!-- source: results/benchmark_summary_v2.csv line 451 --> |
| results/all_121_compressors_gpu_matrix.csv | cur_moe | PASS | Comp Ratio: 1.0x, Time: 0.166s <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 29 --> |
