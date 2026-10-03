# Benchmark Quality & Compression Report: Qwen3.5-0.8B on sst2
- **Execution Timestamp**: `2026-09-06 06:21:51`
- **Baseline Reference Accuracy**: `67.55%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **67.55%** | 0.00% (base) | 0.5501 | 0 | 1.00x | 1810.6 MB | 110.9 |
| **armor_pruner** | `PASS` | **60.21%** | -7.34% | 0.7177 | 0 | 1.00x | 1861.6 MB | 1391.2 |
