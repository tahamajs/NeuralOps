# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_faq/roberta/summary.csv | faq | PASS_NO_COMPRESSION | 80.05% <!-- source: results/unified_faq/roberta/summary.csv line 3 --> | 916.21 MB <!-- source: results/unified_faq/roberta/summary.csv line 3 --> |
| results/unified_faq/distilbert/summary.csv | faq | PASS_NO_COMPRESSION | 90.48% <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> | 416.0 MB <!-- source: results/unified_faq/distilbert/summary.csv line 3 --> |
| results/unified_faq/gpt2/summary.csv | faq | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> | 884.17 MB <!-- source: results/unified_faq/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 26.39 ms <!-- source: results/benchmark_summary_v2.csv line 524 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 524 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.82 ms <!-- source: results/benchmark_summary_v2.csv line 525 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 525 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 26.83 ms <!-- source: results/benchmark_summary_v2.csv line 526 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 526 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 26.56 ms <!-- source: results/benchmark_summary_v2.csv line 527 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 527 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.49 ms <!-- source: results/benchmark_summary_v2.csv line 528 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 528 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.11 ms <!-- source: results/benchmark_summary_v2.csv line 529 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 529 --> |
| results/all_121_compressors_gpu_matrix.csv | faq | PASS | Comp Ratio: 1.0x, Time: 0.243s <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 46 --> |
| results/all_121_compressors_gpu_matrix.csv | faq_quantizer | PASS | Comp Ratio: 1.0x, Time: 0.233s <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 47 --> |
