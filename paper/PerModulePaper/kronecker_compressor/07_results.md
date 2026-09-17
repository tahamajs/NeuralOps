# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_kronecker/roberta/summary.csv | kronecker | PASS | 70.07% <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> | 301.72 MB <!-- source: results/unified_kronecker/roberta/summary.csv line 3 --> |
| results/unified_kronecker/distilbert/summary.csv | kronecker | PASS | 90.94% <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> | 315.07 MB <!-- source: results/unified_kronecker/distilbert/summary.csv line 3 --> |
| results/unified_kronecker/gpt2/summary.csv | kronecker | PASS | 60.32% <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> | 643.3 MB <!-- source: results/unified_kronecker/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | kronecker | PASS | Comp Ratio: 358.64x, Time: 0.782s <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> | 129.8 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 65 --> |
