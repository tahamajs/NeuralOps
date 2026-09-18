# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_spectral_mamba/roberta/summary.csv | spectral_mamba | PASS_LOW | 50.92% <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> | 749.82 MB <!-- source: results/unified_spectral_mamba/roberta/summary.csv line 3 --> |
| results/unified_spectral_mamba/distilbert/summary.csv | spectral_mamba | PASS | 90.71% <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> | 378.15 MB <!-- source: results/unified_spectral_mamba/distilbert/summary.csv line 3 --> |
| results/unified_spectral_mamba/gpt2/summary.csv | spectral_mamba | PASS_LOW | 51.03% <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> | 650.47 MB <!-- source: results/unified_spectral_mamba/gpt2/summary.csv line 3 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 59.34 ms <!-- source: results/benchmark_summary_v2.csv line 224 --> | 1243.5 MB <!-- source: results/benchmark_summary_v2.csv line 224 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 59.26 ms <!-- source: results/benchmark_summary_v2.csv line 225 --> | 1243.5 MB <!-- source: results/benchmark_summary_v2.csv line 225 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 60.69 ms <!-- source: results/benchmark_summary_v2.csv line 226 --> | 1253.4 MB <!-- source: results/benchmark_summary_v2.csv line 226 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 60.06 ms <!-- source: results/benchmark_summary_v2.csv line 227 --> | 1253.4 MB <!-- source: results/benchmark_summary_v2.csv line 227 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 61.36 ms <!-- source: results/benchmark_summary_v2.csv line 228 --> | 1227.8 MB <!-- source: results/benchmark_summary_v2.csv line 228 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 61.24 ms <!-- source: results/benchmark_summary_v2.csv line 229 --> | 1227.8 MB <!-- source: results/benchmark_summary_v2.csv line 229 --> |
