# Benchmark Quality & Compression Report: gpt2 on sst2
- **Execution Timestamp**: `2026-09-06 03:55:30`
- **Baseline Reference Accuracy**: `81.31%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **81.31%** | 0.00% (base) | 0.8372 | 0 | 1.00x | 392.0 MB | 6.1 |
| **adasvd** | `PASS` | **77.06%** | -4.25% | 0.8051 | 24 | 1.07x | 393.5 MB | 88.2 |
