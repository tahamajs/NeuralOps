# 7. Empirical Results

## 7.1 Unified Benchmark Results

| Source File | Configuration | Status | Accuracy / Latency | Peak VRAM |
|---|---|---|---|---|
| results/unified_rotational_slicegpt_int4/roberta/summary.csv | rotational_slicegpt_int4 | PASS | 87.5% <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> | 360.79 MB <!-- source: results/unified_rotational_slicegpt_int4/roberta/summary.csv line 3 --> |
| results/unified_rotational_slicegpt_int4/distilbert/summary.csv | rotational_slicegpt_int4 | PASS | 88.99% <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> | 339.84 MB <!-- source: results/unified_rotational_slicegpt_int4/distilbert/summary.csv line 3 --> |
| results/unified_rotational_slicegpt_int4/gpt2/summary.csv | rotational_slicegpt_int4 | PASS | 61.7% <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> | 636.04 MB <!-- source: results/unified_rotational_slicegpt_int4/gpt2/summary.csv line 3 --> |
| results/all_121_compressors_gpu_matrix.csv | rotational_slicegpt_int4 | PASS | Comp Ratio: 1.0x, Time: 19.574s <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> | 254.5 MB <!-- source: results/all_121_compressors_gpu_matrix.csv line 98 --> |
