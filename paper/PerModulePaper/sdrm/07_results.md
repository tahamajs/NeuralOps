# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_sdrm/roberta/summary.csv | sdrm | PASS_LOW | 50.92% <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> | 781.79 MB <!-- source: results/unified_sdrm/roberta/summary.csv line 3 --> |
| results/unified_sdrm/distilbert/summary.csv | sdrm | PASS | 85.78% <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> | 338.98 MB <!-- source: results/unified_sdrm/distilbert/summary.csv line 3 --> |
| results/unified_sdrm/gpt2/summary.csv | sdrm | PASS | 67.66% <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> | 446.9 MB <!-- source: results/unified_sdrm/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.51 ms <!-- source: results/benchmark_summary_v2.csv line 686 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 686 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.67 ms <!-- source: results/benchmark_summary_v2.csv line 687 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 687 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.54 ms <!-- source: results/benchmark_summary_v2.csv line 688 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 688 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 27.13 ms <!-- source: results/benchmark_summary_v2.csv line 689 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 689 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 53.04 ms <!-- source: results/benchmark_summary_v2.csv line 690 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 690 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.07 ms <!-- source: results/benchmark_summary_v2.csv line 691 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 691 --> |
| results/all_121_compressors_gpu_matrix.csv | sdrm | PASS | Comp Ratio: 1.43x, Time: 0.896s <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> | 149.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 103 --> |
