# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_lsc/roberta/summary.csv | lsc | PASS | 72.02% <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_lsc/roberta/summary.csv line 3 --> |
| results/unified_lsc/distilbert/summary.csv | lsc | PASS | 90.48% <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> | 311.19 MB <!-- source: results/unified_lsc/distilbert/summary.csv line 3 --> |
| results/unified_lsc/gpt2/summary.csv | lsc | PASS | 62.04% <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_lsc/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | lsc | PASS | Comp Ratio: 1.0x, Time: 0.402s <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> | 197.3 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 73 --> |
| results/all_121_compressors_gpu_matrix.csv | lsc_compressor | PASS | Comp Ratio: 1.0x, Time: 0.189s <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> | 197.3 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 74 --> |
