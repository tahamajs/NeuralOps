# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_oats/roberta/summary.csv | oats | PASS | 69.38% <!-- source: results/unified_oats/roberta/summary.csv line 3 --> | 864.62 MB <!-- source: results/unified_oats/roberta/summary.csv line 3 --> |
| results/unified_oats/distilbert/summary.csv | oats | PASS | 90.02% <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> | 226.48 MB <!-- source: results/unified_oats/distilbert/summary.csv line 3 --> |
| results/unified_oats/gpt2/summary.csv | oats | PASS | 76.03% <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> | 856.33 MB <!-- source: results/unified_oats/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.64 ms <!-- source: results/benchmark_summary_v2.csv line 656 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 656 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 26.07 ms <!-- source: results/benchmark_summary_v2.csv line 657 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 657 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.86 ms <!-- source: results/benchmark_summary_v2.csv line 658 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 658 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 27.32 ms <!-- source: results/benchmark_summary_v2.csv line 659 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 659 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 53.03 ms <!-- source: results/benchmark_summary_v2.csv line 660 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 660 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 51.94 ms <!-- source: results/benchmark_summary_v2.csv line 661 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 661 --> |
| results/all_121_compressors_gpu_matrix.csv | oats | PASS | Comp Ratio: 1.0x, Time: 3.735s <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> | 294.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 85 --> |
