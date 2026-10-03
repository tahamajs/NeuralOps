# Phase 2 Quality Benchmark & Statistical Significance Report
**Generated:** preflight_leak_test  
**Hardware Platform:** NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Statistical Protocol:** $n=3$ independent random seeds (`[42, 123, 777]`), 95% Confidence Intervals, Paired Student's t-test vs `full` baseline.  

## Quality & Statistical Significance Summary Table
| Architecture | Model | Dataset | Accuracy (%) | 95% CI | $p$-value vs Base | F1 Score | Fidelity | Fwd (ms) | VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `kmeans` | DISTILBERT | SST2 | **90.14% ± 0.00%** | `[90.1, 90.1]` | 1.0000 | 0.9049 | 0.9147 | 36.02 | 2229 |
| `full` | DISTILBERT | SST2 | **89.33% ± 0.00%** | `[89.3, 89.3]` | -- | 0.8961 | 1.0000 | 33.74 | 2523 |
| `svd_linear_attention` | DISTILBERT | SST2 | **80.73% ± 0.00%** | `[80.7, 80.7]` | 1.0000 | 0.8019 | 0.6199 | 63.42 | 2818 |
| `fused_svd_kernel` | DISTILBERT | SST2 | **80.73% ± 0.00%** | `[80.7, 80.7]` | 1.0000 | 0.8019 | 0.6196 | 344.64 | 2720 |
| `spectron` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.0671 | 47.25 | 2858 |