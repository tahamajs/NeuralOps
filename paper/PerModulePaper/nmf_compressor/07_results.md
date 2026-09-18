# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_nmf/roberta/summary.csv | nmf | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_nmf/roberta/summary.csv line 3 --> |
| results/unified_nmf/distilbert/summary.csv | nmf | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_nmf/distilbert/summary.csv line 3 --> |
| results/unified_nmf/gpt2/summary.csv | nmf | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> | 626.91 MB <!-- source: results/unified_nmf/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | nmf | PASS | Comp Ratio: 30.96x, Time: 9.873s <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> | 101.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 83 --> |
| results/all_121_compressors_gpu_matrix.csv | nmf_compressor | PASS | Comp Ratio: 30.96x, Time: 9.967s <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> | 101.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 84 --> |
