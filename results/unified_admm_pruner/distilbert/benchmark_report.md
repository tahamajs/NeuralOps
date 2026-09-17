# Benchmark Quality & Compression Report: distilbert_sst2 on sst2
- **Execution Timestamp**: `2026-09-06 05:07:23`
- **Baseline Reference Accuracy**: `90.48%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **90.48%** | 0.00% (base) | 0.9077 | 0 | 1.00x | 390.7 MB | 3.1 |
| **admm_pruner** | `PASS` | **84.29%** | -6.19% | 0.8603 | 36 | 1.00x | 551.2 MB | 25.3 |
