# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_kronecker_tt_lora/roberta/summary.csv | kronecker_tt_lora | FAIL | 0.0% <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> | 712.34 MB <!-- source: results/unified_kronecker_tt_lora/roberta/summary.csv line 3 --> |
| results/unified_kronecker_tt_lora/distilbert/summary.csv | kronecker_tt_lora | PASS | 90.48% <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> | 211.47 MB <!-- source: results/unified_kronecker_tt_lora/distilbert/summary.csv line 3 --> |
| results/unified_kronecker_tt_lora/gpt2/summary.csv | kronecker_tt_lora | PASS | 76.95% <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> | 803.66 MB <!-- source: results/unified_kronecker_tt_lora/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 83.84 ms <!-- source: results/benchmark_summary_v2.csv line 104 --> | 761.5 MB <!-- source: results/benchmark_summary_v2.csv line 104 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 87.18 ms <!-- source: results/benchmark_summary_v2.csv line 105 --> | 761.5 MB <!-- source: results/benchmark_summary_v2.csv line 105 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 96.61 ms <!-- source: results/benchmark_summary_v2.csv line 106 --> | 771.4 MB <!-- source: results/benchmark_summary_v2.csv line 106 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 101.79 ms <!-- source: results/benchmark_summary_v2.csv line 107 --> | 771.4 MB <!-- source: results/benchmark_summary_v2.csv line 107 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 99.66 ms <!-- source: results/benchmark_summary_v2.csv line 108 --> | 1394.0 MB <!-- source: results/benchmark_summary_v2.csv line 108 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 99.89 ms <!-- source: results/benchmark_summary_v2.csv line 109 --> | 1394.0 MB <!-- source: results/benchmark_summary_v2.csv line 109 --> |
| results/all_121_compressors_gpu_matrix.csv | kronecker_tt_lora | PASS | Comp Ratio: 0.98x, Time: 0.315s <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> | 269.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 66 --> |
