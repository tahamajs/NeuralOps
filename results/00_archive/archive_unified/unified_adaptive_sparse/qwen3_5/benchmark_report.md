# Benchmark Quality & Compression Report: Qwen3.5-0.8B on sst2
- **Execution Timestamp**: `2026-09-06 23:20:10`
- **Baseline Reference Accuracy**: `67.55%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **67.55%** | 0.00% (base) | 0.5501 | 0 | 1.00x | 1810.6 MB | 107.9 |
| **adaptive_sparse** | **`SKIP_QWEN3_5_ARCHITECTURAL`** | **0.00%** | -67.55% | 0.5501 | 0 | 1.00x | 1810.6 MB | 3.5 |
