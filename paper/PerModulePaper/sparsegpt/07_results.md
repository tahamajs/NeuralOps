# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_sparsegpt/roberta/summary.csv | sparsegpt | PASS_LOW | 50.92% <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> | 363.54 MB <!-- source: results/unified_sparsegpt/roberta/summary.csv line 3 --> |
| results/unified_sparsegpt/distilbert/summary.csv | sparsegpt | PASS | 86.93% <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> | 340.09 MB <!-- source: results/unified_sparsegpt/distilbert/summary.csv line 3 --> |
| results/unified_sparsegpt/gpt2/summary.csv | sparsegpt | PASS_LOW | 53.33% <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> | 634.41 MB <!-- source: results/unified_sparsegpt/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 27.98 ms <!-- source: results/benchmark_summary_v2.csv line 596 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 596 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 31.15 ms <!-- source: results/benchmark_summary_v2.csv line 597 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 597 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 32.37 ms <!-- source: results/benchmark_summary_v2.csv line 598 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 598 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 38.08 ms <!-- source: results/benchmark_summary_v2.csv line 599 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 599 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.92 ms <!-- source: results/benchmark_summary_v2.csv line 600 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 600 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 53.38 ms <!-- source: results/benchmark_summary_v2.csv line 601 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 601 --> |
| results/all_121_compressors_gpu_matrix.csv | sparsegpt | PASS | Comp Ratio: 1.0x, Time: 7.585s <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> | 222.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 111 --> |
