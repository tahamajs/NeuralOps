# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_flat_llm/roberta/summary.csv | flat_llm | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_flat_llm/roberta/summary.csv line 3 --> |
| results/unified_flat_llm/distilbert/summary.csv | flat_llm | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_flat_llm/distilbert/summary.csv line 3 --> |
| results/unified_flat_llm/gpt2/summary.csv | flat_llm | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> | 630.16 MB <!-- source: results/unified_flat_llm/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.69 ms <!-- source: results/benchmark_summary_v2.csv line 632 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 632 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 25.27 ms <!-- source: results/benchmark_summary_v2.csv line 633 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 633 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 27.27 ms <!-- source: results/benchmark_summary_v2.csv line 634 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 634 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 29.01 ms <!-- source: results/benchmark_summary_v2.csv line 635 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 635 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 53.07 ms <!-- source: results/benchmark_summary_v2.csv line 636 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 636 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 51.97 ms <!-- source: results/benchmark_summary_v2.csv line 637 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 637 --> |
| results/all_121_compressors_gpu_matrix.csv | flat_llm | PASS | Comp Ratio: 1.0x, Time: 0.175s <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 50 --> |
