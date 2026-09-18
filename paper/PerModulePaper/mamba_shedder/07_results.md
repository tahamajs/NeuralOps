# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_mamba_shedder/roberta/summary.csv | mamba_shedder | PASS | 85.44% <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_mamba_shedder/roberta/summary.csv line 3 --> |
| results/unified_mamba_shedder/distilbert/summary.csv | mamba_shedder | PASS | 87.5% <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_mamba_shedder/distilbert/summary.csv line 3 --> |
| results/unified_mamba_shedder/gpt2/summary.csv | mamba_shedder | PASS | 56.88% <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> | 634.91 MB <!-- source: results/unified_mamba_shedder/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 25.97 ms <!-- source: results/benchmark_summary_v2.csv line 680 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 680 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 26.73 ms <!-- source: results/benchmark_summary_v2.csv line 681 --> | 571.8 MB <!-- source: results/benchmark_summary_v2.csv line 681 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 28.22 ms <!-- source: results/benchmark_summary_v2.csv line 682 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 682 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 27.66 ms <!-- source: results/benchmark_summary_v2.csv line 683 --> | 581.7 MB <!-- source: results/benchmark_summary_v2.csv line 683 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 52.61 ms <!-- source: results/benchmark_summary_v2.csv line 684 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 684 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 52.19 ms <!-- source: results/benchmark_summary_v2.csv line 685 --> | 1107.2 MB <!-- source: results/benchmark_summary_v2.csv line 685 --> |
| results/all_121_compressors_gpu_matrix.csv | mamba_shedder | PASS | Comp Ratio: 1.0x, Time: 0.173s <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 75 --> |
