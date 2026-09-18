# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_soka/roberta/summary.csv | soka | PASS | 84.06% <!-- source: results/unified_soka/roberta/summary.csv line 3 --> | 359.04 MB <!-- source: results/unified_soka/roberta/summary.csv line 3 --> |
| results/unified_soka/distilbert/summary.csv | soka | PASS | 89.91% <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> | 339.22 MB <!-- source: results/unified_soka/distilbert/summary.csv line 3 --> |
| results/unified_soka/gpt2/summary.csv | soka | PASS | 60.78% <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> | 637.1 MB <!-- source: results/unified_soka/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 34.47 ms <!-- source: results/benchmark_summary_v2.csv line 668 --> | 711.7 MB <!-- source: results/benchmark_summary_v2.csv line 668 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 33.71 ms <!-- source: results/benchmark_summary_v2.csv line 669 --> | 711.7 MB <!-- source: results/benchmark_summary_v2.csv line 669 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 36.18 ms <!-- source: results/benchmark_summary_v2.csv line 670 --> | 721.6 MB <!-- source: results/benchmark_summary_v2.csv line 670 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 36.11 ms <!-- source: results/benchmark_summary_v2.csv line 671 --> | 721.6 MB <!-- source: results/benchmark_summary_v2.csv line 671 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 49.91 ms <!-- source: results/benchmark_summary_v2.csv line 672 --> | 1077.4 MB <!-- source: results/benchmark_summary_v2.csv line 672 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.14 ms <!-- source: results/benchmark_summary_v2.csv line 673 --> | 1077.4 MB <!-- source: results/benchmark_summary_v2.csv line 673 --> |
| results/all_121_compressors_gpu_matrix.csv | soka | PASS | Comp Ratio: 59.91x, Time: 0.756s <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> | 136.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 109 --> |
