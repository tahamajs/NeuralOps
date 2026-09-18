# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_liftr/roberta/summary.csv | liftr | PASS | 80.16% <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> | 585.62 MB <!-- source: results/unified_liftr/roberta/summary.csv line 3 --> |
| results/unified_liftr/distilbert/summary.csv | liftr | PASS | 90.14% <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_liftr/distilbert/summary.csv line 3 --> |
| results/unified_liftr/gpt2/summary.csv | liftr | PASS | 61.01% <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_liftr/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 27.04 ms <!-- source: results/benchmark_summary_v2.csv line 674 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 674 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.29 ms <!-- source: results/benchmark_summary_v2.csv line 675 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 675 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 27.23 ms <!-- source: results/benchmark_summary_v2.csv line 676 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 676 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 29.18 ms <!-- source: results/benchmark_summary_v2.csv line 677 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 677 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 17.91 ms <!-- source: results/benchmark_summary_v2.csv line 678 --> | 429.4 MB <!-- source: results/benchmark_summary_v2.csv line 678 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 18.19 ms <!-- source: results/benchmark_summary_v2.csv line 679 --> | 429.4 MB <!-- source: results/benchmark_summary_v2.csv line 679 --> |
| results/all_121_compressors_gpu_matrix.csv | liftr | PASS | Comp Ratio: 1.0x, Time: 0.169s <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 69 --> |
