# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_frandi/roberta/summary.csv | frandi | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_frandi/roberta/summary.csv line 3 --> | 520.68 MB <!-- source: results/unified_frandi/roberta/summary.csv line 3 --> |
| results/unified_frandi/distilbert/summary.csv | frandi | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_frandi/distilbert/summary.csv line 3 --> | 301.18 MB <!-- source: results/unified_frandi/distilbert/summary.csv line 3 --> |
| results/unified_frandi/gpt2/summary.csv | frandi | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_frandi/gpt2/summary.csv line 3 --> | 627.16 MB <!-- source: results/unified_frandi/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 35.36 ms <!-- source: results/benchmark_summary_v2.csv line 146 --> | 581.9 MB <!-- source: results/benchmark_summary_v2.csv line 146 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 34.99 ms <!-- source: results/benchmark_summary_v2.csv line 147 --> | 581.9 MB <!-- source: results/benchmark_summary_v2.csv line 147 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 35.5 ms <!-- source: results/benchmark_summary_v2.csv line 148 --> | 591.9 MB <!-- source: results/benchmark_summary_v2.csv line 148 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 35.21 ms <!-- source: results/benchmark_summary_v2.csv line 149 --> | 591.9 MB <!-- source: results/benchmark_summary_v2.csv line 149 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 49.9 ms <!-- source: results/benchmark_summary_v2.csv line 150 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 150 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 49.79 ms <!-- source: results/benchmark_summary_v2.csv line 151 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 151 --> |
| results/all_121_compressors_gpu_matrix.csv | frandi | PASS | Comp Ratio: 30.96x, Time: 0.791s <!-- source: results/all_121_compressors_gpu_matrix.csv line 52 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 52 --> |
