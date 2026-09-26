# Benchmark Quality & Compression Report: gpt2 on sst2
- **Execution Timestamp**: `2026-09-06 05:12:06`
- **Baseline Reference Accuracy**: `81.31%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **81.31%** | 0.00% (base) | 0.8372 | 0 | 1.00x | 392.0 MB | 6.3 |
| **admm_pruner** | `PASS` | **80.73%** | -0.58% | 0.8146 | 48 | 1.00x | 414.9 MB | 83.7 |
