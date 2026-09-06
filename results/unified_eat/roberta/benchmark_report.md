# Benchmark Quality & Compression Report: roberta-base on sst2
- **Execution Timestamp**: `2026-09-07 03:01:54`
- **Baseline Reference Accuracy**: `88.76%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **88.76%** | 0.00% (base) | 0.8942 | 0 | 1.00x | 300.7 MB | 6.0 |
| **eat** | `PASS` | **90.60%** | +1.84% | 0.9113 | 0 | 0.95x | 400.0 MB | 122.3 |
