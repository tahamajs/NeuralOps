# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_nli_softmax/roberta/summary.csv | nli_softmax | PASS | 60.32% <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> | 640.61 MB <!-- source: results/unified_nli_softmax/roberta/summary.csv line 3 --> |
| results/unified_nli_softmax/distilbert/summary.csv | nli_softmax | PASS | 89.33% <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> | 315.91 MB <!-- source: results/unified_nli_softmax/distilbert/summary.csv line 3 --> |
| results/unified_nli_softmax/gpt2/summary.csv | nli_softmax | PASS_LOW | 55.28% <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> | 557.88 MB <!-- source: results/unified_nli_softmax/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | nli_softmax | PASS | Comp Ratio: 1.0x, Time: 0.166s <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> | 127.0 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 81 --> |
