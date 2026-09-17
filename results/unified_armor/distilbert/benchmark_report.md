# Benchmark Quality & Compression Report: distilbert_sst2 on sst2
- **Execution Timestamp**: `2026-09-06 05:51:02`
- **Baseline Reference Accuracy**: `90.48%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **90.48%** | 0.00% (base) | 0.9077 | 0 | 1.00x | 390.7 MB | 3.1 |
| **armor_pruner** | `PASS` | **89.68%** | -0.80% | 0.9002 | 0 | 1.00x | 311.9 MB | 15.5 |
