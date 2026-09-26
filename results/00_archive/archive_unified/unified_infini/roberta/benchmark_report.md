# Benchmark Quality & Compression Report: roberta-base on sst2
- **Execution Timestamp**: `2026-09-07 05:00:11`
- **Baseline Reference Accuracy**: `88.76%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **88.76%** | 0.00% (base) | 0.8942 | 0 | 1.00x | 299.9 MB | 6.6 |
| **infini** | `PASS` | **83.60%** | -5.16% | 0.8566 | 0 | 0.94x | 405.6 MB | 31.0 |
