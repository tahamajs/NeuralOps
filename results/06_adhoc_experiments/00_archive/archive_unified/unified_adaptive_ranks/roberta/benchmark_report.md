# Benchmark Quality & Compression Report: roberta-base on sst2
- **Execution Timestamp**: `2026-09-06 03:17:22`
- **Baseline Reference Accuracy**: `58.83%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **adaptive_ranks** | `PASS` | **65.02%** | +6.19% | 0.4925 | 48 | 1.19x | 612.8 MB | 40.3 |
