# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_zerokv_neo/roberta/summary.csv | zerokv_neo | PASS_LOW | 50.92% <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> | 531.69 MB <!-- source: results/unified_zerokv_neo/roberta/summary.csv line 3 --> |
| results/unified_zerokv_neo/distilbert/summary.csv | zerokv_neo | PASS | 75.92% <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> | 312.19 MB <!-- source: results/unified_zerokv_neo/distilbert/summary.csv line 3 --> |
| results/unified_zerokv_neo/gpt2/summary.csv | zerokv_neo | PASS_LOW | 51.15% <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> | 413.5 MB <!-- source: results/unified_zerokv_neo/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 43.31 ms <!-- source: results/benchmark_summary_v2.csv line 542 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 542 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 22.97 ms <!-- source: results/benchmark_summary_v2.csv line 543 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 543 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 22.55 ms <!-- source: results/benchmark_summary_v2.csv line 544 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 544 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 24.82 ms <!-- source: results/benchmark_summary_v2.csv line 545 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 545 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 51.97 ms <!-- source: results/benchmark_summary_v2.csv line 546 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 546 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 51.84 ms <!-- source: results/benchmark_summary_v2.csv line 547 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 547 --> |
| results/all_121_compressors_gpu_matrix.csv | zerokv_neo | PASS | Comp Ratio: 1.0x, Time: 0.177s <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 122 --> |
