# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_galerkin_attention/roberta/summary.csv | galerkin_attention | PASS_NO_COMPRESSION | 49.2% <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> | 653.87 MB <!-- source: results/unified_galerkin_attention/roberta/summary.csv line 3 --> |
| results/unified_galerkin_attention/distilbert/summary.csv | galerkin_attention | PASS_NO_COMPRESSION | 65.94% <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> | 322.36 MB <!-- source: results/unified_galerkin_attention/distilbert/summary.csv line 3 --> |
| results/unified_galerkin_attention/gpt2/summary.csv | galerkin_attention | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> | 567.91 MB <!-- source: results/unified_galerkin_attention/gpt2/summary.csv line 3 --> |
| results/agnews_unified_galerkin_attention/distilbert/summary.csv | galerkin_attention | PASS_NO_COMPRESSION | 36.0% <!-- source: results/agnews_unified_galerkin_attention/distilbert/summary.csv line 2 --> | 541.99 MB <!-- source: results/agnews_unified_galerkin_attention/distilbert/summary.csv line 2 --> |
| results/mnli_unified_galerkin_attention/distilbert/summary.csv | galerkin_attention | PASS_NO_COMPRESSION | 36.4% <!-- source: results/mnli_unified_galerkin_attention/distilbert/summary.csv line 2 --> | 356.28 MB <!-- source: results/mnli_unified_galerkin_attention/distilbert/summary.csv line 2 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 36.59 ms <!-- source: results/benchmark_summary_v2.csv line 308 --> | 497.4 MB <!-- source: results/benchmark_summary_v2.csv line 308 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 36.94 ms <!-- source: results/benchmark_summary_v2.csv line 309 --> | 497.4 MB <!-- source: results/benchmark_summary_v2.csv line 309 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 38.29 ms <!-- source: results/benchmark_summary_v2.csv line 310 --> | 507.3 MB <!-- source: results/benchmark_summary_v2.csv line 310 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 38.06 ms <!-- source: results/benchmark_summary_v2.csv line 311 --> | 507.3 MB <!-- source: results/benchmark_summary_v2.csv line 311 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 37.62 ms <!-- source: results/benchmark_summary_v2.csv line 312 --> | 580.8 MB <!-- source: results/benchmark_summary_v2.csv line 312 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 36.94 ms <!-- source: results/benchmark_summary_v2.csv line 313 --> | 580.8 MB <!-- source: results/benchmark_summary_v2.csv line 313 --> |
