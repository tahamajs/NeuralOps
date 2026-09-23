# Benchmark Quality & Compression Report: Qwen3.5-0.8B on sst2
- **Execution Timestamp**: `2026-09-06 04:51:58`
- **Baseline Reference Accuracy**: `67.55%`
- **Statistical Pass Threshold**: `56.00%`

## Empirical Comparative Profiling

| Module | Status | Accuracy | Delta Acc | F1 Score | Compressed Layers | Comp Rate | Peak VRAM | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **full** | `PASS` | **67.55%** | 0.00% (base) | 0.5501 | 0 | 1.00x | 1810.6 MB | 111.0 |
| **admm** | `PASS` | **88.42%** | +20.87% | 0.8947 | 28 | 1.00x | 1851.3 MB | 1355.4 |
