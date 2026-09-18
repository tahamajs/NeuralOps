# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_slicegpt_pca/roberta/summary.csv | slicegpt_pca | PASS | 67.66% <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> | 300.97 MB <!-- source: results/unified_slicegpt_pca/roberta/summary.csv line 3 --> |
| results/unified_slicegpt_pca/distilbert/summary.csv | slicegpt_pca | PASS | 90.48% <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> | 312.19 MB <!-- source: results/unified_slicegpt_pca/distilbert/summary.csv line 3 --> |
| results/unified_slicegpt_pca/gpt2/summary.csv | slicegpt_pca | PASS | 62.04% <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> | 637.29 MB <!-- source: results/unified_slicegpt_pca/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.98 ms <!-- source: results/benchmark_summary_v2.csv line 590 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 590 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.11 ms <!-- source: results/benchmark_summary_v2.csv line 591 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 591 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 28.27 ms <!-- source: results/benchmark_summary_v2.csv line 592 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 592 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 27.03 ms <!-- source: results/benchmark_summary_v2.csv line 593 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 593 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.62 ms <!-- source: results/benchmark_summary_v2.csv line 594 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 594 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.0 ms <!-- source: results/benchmark_summary_v2.csv line 595 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 595 --> |
| results/all_121_compressors_gpu_matrix.csv | slicegpt_pca | PASS | Comp Ratio: 1.0x, Time: 58.804s <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> | 135.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 106 --> |
