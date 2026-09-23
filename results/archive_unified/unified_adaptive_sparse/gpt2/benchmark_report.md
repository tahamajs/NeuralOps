# Benchmark Quality & Compression Report: gpt2 on sst2
- **Execution Timestamp**: `2026-09-06 23:15:51`
- **Baseline Reference Accuracy**: `81.31%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **81.31%** | 0.00% (base) | 0.8372 | 0 | 1.00x | 392.0 MB | 6.1 |
| **adaptive_sparse** | **`SKIP_GPT_ARCHITECTURAL`** | **0.00%** | -81.31% | 0.8372 | 0 | 1.00x | 392.0 MB | 1.1 |
