# Benchmark Quality & Compression Report: distilbert_sst2 on sst2
- **Execution Timestamp**: `2026-09-06 03:50:30`
- **Baseline Reference Accuracy**: `90.48%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **90.48%** | 0.00% (base) | 0.9077 | 0 | 1.00x | 390.7 MB | 3.1 |
| **adasvd** | `PASS` | **84.86%** | -5.62% | 0.8514 | 24 | 1.06x | 413.2 MB | 34.7 |
