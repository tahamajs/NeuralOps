# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_optimal_subspace_projection/roberta/summary.csv | optimal_subspace_projection | PASS | 82.34% <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> | 556.31 MB <!-- source: results/unified_optimal_subspace_projection/roberta/summary.csv line 3 --> |
| results/unified_optimal_subspace_projection/distilbert/summary.csv | optimal_subspace_projection | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> | 183.07 MB <!-- source: results/unified_optimal_subspace_projection/distilbert/summary.csv line 3 --> |
| results/unified_optimal_subspace_projection/gpt2/summary.csv | optimal_subspace_projection | PASS | 78.1% <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> | 644.16 MB <!-- source: results/unified_optimal_subspace_projection/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | optimal_subspace_projection | PASS | Comp Ratio: 1.0x, Time: 0.44s <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> | 204.4 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 87 --> |
