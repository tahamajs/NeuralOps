# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_subspace_projection/roberta/summary.csv | subspace_projection | PASS | 84.17% <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> | 364.29 MB <!-- source: results/unified_subspace_projection/roberta/summary.csv line 3 --> |
| results/unified_subspace_projection/distilbert/summary.csv | subspace_projection | PASS | 89.91% <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> | 338.84 MB <!-- source: results/unified_subspace_projection/distilbert/summary.csv line 3 --> |
| results/unified_subspace_projection/gpt2/summary.csv | subspace_projection | PASS | 59.52% <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> | 637.29 MB <!-- source: results/unified_subspace_projection/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | subspace_projection | PASS | Comp Ratio: 30.96x, Time: 0.757s <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> | 115.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 113 --> |
