# Benchmark Quality & Compression Report: roberta-base on sst2
- **Execution Timestamp**: `2026-09-06 03:38:07`
- **Baseline Reference Accuracy**: `88.76%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **aasvd** | `PASS` | **90.02%** | +1.26% | 0.9024 | 48 | 1.06x | 318.4 MB | 109.7 |
| **full** | `PASS` | **88.76%** | 0.00% (base) | 0.8942 | 0 | 1.00x | 300.7 MB | 5.8 |
