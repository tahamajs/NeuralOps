# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_come/roberta/summary.csv | come | FAIL | 0.0% <!-- source: results/unified_come/roberta/summary.csv line 3 --> | 712.34 MB <!-- source: results/unified_come/roberta/summary.csv line 3 --> |
| results/unified_come/distilbert/summary.csv | come | PASS | 84.29% <!-- source: results/unified_come/distilbert/summary.csv line 3 --> | 259.48 MB <!-- source: results/unified_come/distilbert/summary.csv line 3 --> |
| results/unified_come/qwen3_5/summary.csv | come | FAIL | 0.0% <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> | 1796.78 MB <!-- source: results/unified_come/qwen3_5/summary.csv line 3 --> |
| results/unified_come/gpt2/summary.csv | come | FAIL | 0.0% <!-- source: results/unified_come/gpt2/summary.csv line 3 --> | 793.19 MB <!-- source: results/unified_come/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 16.58 ms <!-- source: results/benchmark_summary_v2.csv line 650 --> | 377.7 MB <!-- source: results/benchmark_summary_v2.csv line 650 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 17.18 ms <!-- source: results/benchmark_summary_v2.csv line 651 --> | 377.7 MB <!-- source: results/benchmark_summary_v2.csv line 651 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 18.51 ms <!-- source: results/benchmark_summary_v2.csv line 652 --> | 387.6 MB <!-- source: results/benchmark_summary_v2.csv line 652 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 18.26 ms <!-- source: results/benchmark_summary_v2.csv line 653 --> | 387.6 MB <!-- source: results/benchmark_summary_v2.csv line 653 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 30.43 ms <!-- source: results/benchmark_summary_v2.csv line 654 --> | 721.9 MB <!-- source: results/benchmark_summary_v2.csv line 654 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 29.4 ms <!-- source: results/benchmark_summary_v2.csv line 655 --> | 721.9 MB <!-- source: results/benchmark_summary_v2.csv line 655 --> |
| results/all_121_compressors_gpu_matrix.csv | come | PASS | Comp Ratio: 2.0x, Time: 0.161s <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> | 72.9 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 23 --> |
