# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_slim/roberta/summary.csv | slim | PASS_LOW | 50.92% <!-- source: results/unified_slim/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_slim/roberta/summary.csv line 3 --> |
| results/unified_slim/distilbert/summary.csv | slim | PASS | 87.96% <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> | 339.21 MB <!-- source: results/unified_slim/distilbert/summary.csv line 3 --> |
| results/unified_slim/gpt2/summary.csv | slim | PASS_LOW | 52.41% <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> | 640.66 MB <!-- source: results/unified_slim/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | slim | PASS | Comp Ratio: 1.0x, Time: 1.028s <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> | 196.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 107 --> |
| results/all_121_compressors_gpu_matrix.csv | slim_compressor | PASS | Comp Ratio: 1.0x, Time: 0.781s <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> | 196.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 108 --> |
