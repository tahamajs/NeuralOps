# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_uniql/roberta/summary.csv | uniql | PASS | 82.45% <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> | 359.04 MB <!-- source: results/unified_uniql/roberta/summary.csv line 3 --> |
| results/unified_uniql/distilbert/summary.csv | uniql | PASS | 90.14% <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> | 339.22 MB <!-- source: results/unified_uniql/distilbert/summary.csv line 3 --> |
| results/unified_uniql/gpt2/summary.csv | uniql | PASS | 60.89% <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> | 635.04 MB <!-- source: results/unified_uniql/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | uniql | PASS | Comp Ratio: 1.0x, Time: 0.813s <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> | 128.5 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 120 --> |
| results/all_121_compressors_gpu_matrix.csv | uniql_compressor | PASS | Comp Ratio: 1.0x, Time: 0.81s <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> | 127.7 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 121 --> |
