# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_lillama/roberta/summary.csv | lillama | PASS | 85.44% <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> | 583.49 MB <!-- source: results/unified_lillama/roberta/summary.csv line 3 --> |
| results/unified_lillama/distilbert/summary.csv | lillama | PASS | 87.5% <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_lillama/distilbert/summary.csv line 3 --> |
| results/unified_lillama/gpt2/summary.csv | lillama | PASS | 56.88% <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_lillama/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.96 ms <!-- source: results/benchmark_summary_v2.csv line 638 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 638 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.44 ms <!-- source: results/benchmark_summary_v2.csv line 639 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 639 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.47 ms <!-- source: results/benchmark_summary_v2.csv line 640 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 640 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 28.13 ms <!-- source: results/benchmark_summary_v2.csv line 641 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 641 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.5 ms <!-- source: results/benchmark_summary_v2.csv line 642 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 642 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.27 ms <!-- source: results/benchmark_summary_v2.csv line 643 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 643 --> |
| results/all_121_compressors_gpu_matrix.csv | lillama | PASS | Comp Ratio: 1.0x, Time: 0.406s <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> | 129.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 71 --> |
| results/all_121_compressors_gpu_matrix.csv | lillama_compressor | PASS | Comp Ratio: 1.0x, Time: 0.426s <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> | 129.2 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 72 --> |
