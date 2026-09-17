# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_halo_regla/roberta/summary.csv | halo_regla | PASS_NO_COMPRESSION | 81.31% <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> | 672.44 MB <!-- source: results/unified_halo_regla/roberta/summary.csv line 3 --> |
| results/unified_halo_regla/distilbert/summary.csv | halo_regla | PASS_NO_COMPRESSION | 90.37% <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> | 343.31 MB <!-- source: results/unified_halo_regla/distilbert/summary.csv line 3 --> |
| results/unified_halo_regla/gpt2/summary.csv | halo_regla | PASS_NO_COMPRESSION | 79.59% <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> | 1134.78 MB <!-- source: results/unified_halo_regla/gpt2/summary.csv line 3 --> |
| results/agnews_unified_halo_regla/distilbert/summary.csv | halo_regla | PASS_NO_COMPRESSION | 76.5% <!-- source: results/agnews_unified_halo_regla/distilbert/summary.csv line 2 --> | 685.93 MB <!-- source: results/agnews_unified_halo_regla/distilbert/summary.csv line 2 --> |
| results/mnli_unified_halo_regla/distilbert/summary.csv | halo_regla | PASS_NO_COMPRESSION | 41.4% <!-- source: results/mnli_unified_halo_regla/distilbert/summary.csv line 2 --> | 390.01 MB <!-- source: results/mnli_unified_halo_regla/distilbert/summary.csv line 2 --> |
| results/benchmark_summary_v2.csv | bert-sst2 | PASS | Step Latency: 61.05 ms <!-- source: results/benchmark_summary_v2.csv line 290 --> | 1250.2 MB <!-- source: results/benchmark_summary_v2.csv line 290 --> |
| results/benchmark_summary_v2.csv | bert-mrpc | PASS | Step Latency: 61.74 ms <!-- source: results/benchmark_summary_v2.csv line 291 --> | 1250.2 MB <!-- source: results/benchmark_summary_v2.csv line 291 --> |
| results/benchmark_summary_v2.csv | roberta-sst2 | PASS | Step Latency: 64.6 ms <!-- source: results/benchmark_summary_v2.csv line 292 --> | 1260.1 MB <!-- source: results/benchmark_summary_v2.csv line 292 --> |
| results/benchmark_summary_v2.csv | roberta-mrpc | PASS | Step Latency: 64.28 ms <!-- source: results/benchmark_summary_v2.csv line 293 --> | 1260.1 MB <!-- source: results/benchmark_summary_v2.csv line 293 --> |
| results/benchmark_summary_v2.csv | t5-sst2 | PASS | Step Latency: 65.24 ms <!-- source: results/benchmark_summary_v2.csv line 294 --> | 1218.4 MB <!-- source: results/benchmark_summary_v2.csv line 294 --> |
| results/benchmark_summary_v2.csv | t5-mrpc | PASS | Step Latency: 65.04 ms <!-- source: results/benchmark_summary_v2.csv line 295 --> | 1218.4 MB <!-- source: results/benchmark_summary_v2.csv line 295 --> |
