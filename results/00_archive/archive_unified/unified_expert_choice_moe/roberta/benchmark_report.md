# Benchmark Quality & Compression Report: roberta-base on sst2
- **Execution Timestamp**: `2026-09-07 02:56:53`
- **Baseline Reference Accuracy**: `88.76%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **88.76%** | 0.00% (base) | 0.8942 | 0 | 1.00x | 300.7 MB | 6.0 |
| **expert_choice_moe** | `PASS` | **90.14%** | +1.38% | 0.9079 | 0 | 0.35x | 842.6 MB | 258.3 |
