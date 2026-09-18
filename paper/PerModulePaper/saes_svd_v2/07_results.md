# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_saes_svd_v2/roberta/summary.csv | saes_svd_v2 | PASS | 85.44% <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> | 364.29 MB <!-- source: results/unified_saes_svd_v2/roberta/summary.csv line 3 --> |
| results/unified_saes_svd_v2/distilbert/summary.csv | saes_svd_v2 | PASS | 87.5% <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> | 338.84 MB <!-- source: results/unified_saes_svd_v2/distilbert/summary.csv line 3 --> |
| results/unified_saes_svd_v2/gpt2/summary.csv | saes_svd_v2 | PASS | 56.88% <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> | 635.79 MB <!-- source: results/unified_saes_svd_v2/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 35.9 ms <!-- source: results/benchmark_summary_v2.csv line 110 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 110 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 35.97 ms <!-- source: results/benchmark_summary_v2.csv line 111 --> | 717.9 MB <!-- source: results/benchmark_summary_v2.csv line 111 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 37.2 ms <!-- source: results/benchmark_summary_v2.csv line 112 --> | 727.8 MB <!-- source: results/benchmark_summary_v2.csv line 112 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 36.01 ms <!-- source: results/benchmark_summary_v2.csv line 113 --> | 727.8 MB <!-- source: results/benchmark_summary_v2.csv line 113 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 50.03 ms <!-- source: results/benchmark_summary_v2.csv line 114 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 114 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 49.91 ms <!-- source: results/benchmark_summary_v2.csv line 115 --> | 1082.2 MB <!-- source: results/benchmark_summary_v2.csv line 115 --> |
| results/all_121_compressors_gpu_matrix.csv | saes_svd_v2 | PASS | Comp Ratio: 30.96x, Time: 0.891s <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> | 112.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 101 --> |
