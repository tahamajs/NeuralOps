# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_slicegpt/roberta/summary.csv | slicegpt | PASS_LOW | 50.92% <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> | 366.79 MB <!-- source: results/unified_slicegpt/roberta/summary.csv line 3 --> |
| results/unified_slicegpt/distilbert/summary.csv | slicegpt | PASS | 80.62% <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> | 339.71 MB <!-- source: results/unified_slicegpt/distilbert/summary.csv line 3 --> |
| results/unified_slicegpt/gpt2/summary.csv | slicegpt | PASS | 59.06% <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> | 637.16 MB <!-- source: results/unified_slicegpt/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.77 ms <!-- source: results/benchmark_summary_v2.csv line 584 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 584 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 27.4 ms <!-- source: results/benchmark_summary_v2.csv line 585 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 585 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.9 ms <!-- source: results/benchmark_summary_v2.csv line 586 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 586 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 26.83 ms <!-- source: results/benchmark_summary_v2.csv line 587 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 587 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.42 ms <!-- source: results/benchmark_summary_v2.csv line 588 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 588 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.15 ms <!-- source: results/benchmark_summary_v2.csv line 589 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 589 --> |
| results/all_121_compressors_gpu_matrix.csv | slicegpt | PASS | Comp Ratio: 1.0x, Time: 0.284s <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> | 207.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 105 --> |
