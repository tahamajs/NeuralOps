# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_flap/roberta/summary.csv | flap | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_flap/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_flap/roberta/summary.csv line 3 --> |
| results/unified_flap/distilbert/summary.csv | flap | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_flap/distilbert/summary.csv line 3 --> |
| results/unified_flap/gpt2/summary.csv | flap | PASS | 77.87% <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> | 626.91 MB <!-- source: results/unified_flap/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.54 ms <!-- source: results/benchmark_summary_v2.csv line 626 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 626 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 27.02 ms <!-- source: results/benchmark_summary_v2.csv line 627 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 627 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.88 ms <!-- source: results/benchmark_summary_v2.csv line 628 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 628 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 26.82 ms <!-- source: results/benchmark_summary_v2.csv line 629 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 629 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.8 ms <!-- source: results/benchmark_summary_v2.csv line 630 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 630 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 51.42 ms <!-- source: results/benchmark_summary_v2.csv line 631 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 631 --> |
| results/all_121_compressors_gpu_matrix.csv | flap | PASS | Comp Ratio: 1.0x, Time: 0.187s <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 48 --> |
