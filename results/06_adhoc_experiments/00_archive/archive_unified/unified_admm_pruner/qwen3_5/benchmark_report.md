# Benchmark Quality & Compression Report: Qwen3.5-0.8B on sst2
- **Execution Timestamp**: `2026-09-06 05:39:27`
- **Baseline Reference Accuracy**: `67.55%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **67.55%** | 0.00% (base) | 0.5501 | 0 | 1.00x | 1810.6 MB | 109.9 |
| **admm_pruner** | `PASS` | **78.21%** | +10.66% | 0.8201 | 48 | 1.00x | 1879.8 MB | 1382.3 |
