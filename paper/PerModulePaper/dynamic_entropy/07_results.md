# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_dynamic_entropy/roberta/summary.csv | dynamic_entropy_svd | PASS_NO_COMPRESSION | 78.56% <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 3 --> | 520.18 MB <!-- source: results/unified_dynamic_entropy/roberta/summary.csv line 3 --> |
| results/unified_dynamic_entropy/distilbert/summary.csv | dynamic_entropy_svd | PASS_NO_COMPRESSION | 90.71% <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 3 --> | 300.68 MB <!-- source: results/unified_dynamic_entropy/distilbert/summary.csv line 3 --> |
| results/unified_dynamic_entropy/qwen3_5/summary.csv | dynamic_entropy_svd | PASS | 89.22% <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 3 --> | 1850.64 MB <!-- source: results/unified_dynamic_entropy/qwen3_5/summary.csv line 3 --> |
| results/unified_dynamic_entropy/gpt2/summary.csv | dynamic_entropy_svd | PASS_NO_COMPRESSION | 79.36% <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 3 --> | 627.16 MB <!-- source: results/unified_dynamic_entropy/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | dynamic_entropy | PASS | Comp Ratio: 30.96x, Time: 1.029s <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> | 104.9 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 37 --> |
