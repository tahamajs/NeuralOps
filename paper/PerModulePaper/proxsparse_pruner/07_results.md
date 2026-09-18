# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_proxsparse_pruner/roberta/summary.csv | proxsparse_pruner | PASS_LOW | 50.92% <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_proxsparse_pruner/roberta/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/distilbert/summary.csv | proxsparse_pruner | PASS | 78.33% <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_proxsparse_pruner/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse_pruner/gpt2/summary.csv | proxsparse_pruner | PASS_LOW | 54.01% <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_proxsparse_pruner/gpt2/summary.csv line 3 --> |
| results/unified_proxsparse/roberta/summary.csv | proxsparse | PASS_LOW | 50.92% <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> | 361.54 MB <!-- source: results/unified_proxsparse/roberta/summary.csv line 3 --> |
| results/unified_proxsparse/distilbert/summary.csv | proxsparse | PASS | 78.33% <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> | 337.59 MB <!-- source: results/unified_proxsparse/distilbert/summary.csv line 3 --> |
| results/unified_proxsparse/gpt2/summary.csv | proxsparse | PASS_LOW | 54.01% <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_proxsparse/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | proxsparse | PASS | Comp Ratio: 1.0x, Time: 0.334s <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> | 155.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 88 --> |
| results/all_121_compressors_gpu_matrix.csv | proxsparse_pruner | PASS | Comp Ratio: 1.0x, Time: 0.314s <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> | 155.1 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 90 --> |
