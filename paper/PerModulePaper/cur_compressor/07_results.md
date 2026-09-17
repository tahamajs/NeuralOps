# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_cur/roberta/summary.csv | cur | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_cur/roberta/summary.csv line 3 --> | 519.68 MB <!-- source: results/unified_cur/roberta/summary.csv line 3 --> |
| results/unified_cur/distilbert/summary.csv | cur | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_cur/distilbert/summary.csv line 3 --> |
| results/unified_cur/gpt2/summary.csv | cur | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> | 627.28 MB <!-- source: results/unified_cur/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | cur | PASS | Comp Ratio: 30.75x, Time: 1.041s <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> | 106.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 28 --> |
