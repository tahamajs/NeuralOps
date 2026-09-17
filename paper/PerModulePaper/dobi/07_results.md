# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_dobi/roberta/summary.csv | dobi_svd | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_dobi/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_dobi/roberta/summary.csv line 3 --> |
| results/unified_dobi/distilbert/summary.csv | dobi_svd | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_dobi/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_dobi/distilbert/summary.csv line 3 --> |
| results/unified_dobi/qwen3_5/summary.csv | dobi_svd | PASS | 89.45% <!-- source: results/unified_dobi/qwen3_5/summary.csv line 3 --> | 1850.64 MB <!-- source: results/unified_dobi/qwen3_5/summary.csv line 3 --> |
| results/unified_dobi/gpt2/summary.csv | dobi_svd | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_dobi/gpt2/summary.csv line 3 --> | 627.16 MB <!-- source: results/unified_dobi/gpt2/summary.csv line 3 --> |
| results/agnews_unified_dobi/distilbert/summary.csv | dobi | PASS_NO_COMPRESSION | 70.7% <!-- source: results/agnews_unified_dobi/distilbert/summary.csv line 2 --> | 449.77 MB <!-- source: results/agnews_unified_dobi/distilbert/summary.csv line 2 --> |
| results/mnli_unified_dobi/distilbert/summary.csv | dobi | PASS_NO_COMPRESSION | 41.1% <!-- source: results/mnli_unified_dobi/distilbert/summary.csv line 2 --> | 319.55 MB <!-- source: results/mnli_unified_dobi/distilbert/summary.csv line 2 --> |
| results/all_121_compressors_gpu_matrix.csv | dobi | PASS | Comp Ratio: 30.95x, Time: 0.794s <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> | 106.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 35 --> |
