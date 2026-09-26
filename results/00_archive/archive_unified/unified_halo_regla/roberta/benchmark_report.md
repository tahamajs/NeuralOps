# Benchmark Quality & Compression Report: roberta-base on sst2
- **Execution Timestamp**: `2026-09-07 04:34:07`
- **Baseline Reference Accuracy**: `88.76%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **88.76%** | 0.00% (base) | 0.8942 | 0 | 1.00x | 299.9 MB | 6.0 |
| **halo_regla** | `PASS` | **87.84%** | -0.92% | 0.8896 | 0 | 0.94x | 400.5 MB | 34.7 |
