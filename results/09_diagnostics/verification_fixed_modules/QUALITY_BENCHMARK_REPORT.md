# Phase 2 Quality Benchmark & Statistical Significance Report
**Generated:** verification_fixed_modules  
**Hardware Platform:** NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Statistical Protocol:** $n=3$ independent random seeds (`[42, 123, 777]`), 95% Confidence Intervals, Paired Student's t-test vs `full` baseline.  

## Quality & Statistical Significance Summary Table
| Architecture | Model | Dataset | Accuracy (%) | 95% CI | $p$-value vs Base | F1 Score | Fidelity | Fwd (ms) | VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full` | DISTILBERT | SST2 | **89.33% ± 0.00%** | `[89.3, 89.3]` | -- | 0.8961 | 1.0000 | 33.83 | 2523 |
| `gated_linear_attention` | DISTILBERT | SST2 | **78.56% ± 0.00%** | `[78.6, 78.6]` | 1.0000 | 0.7797 | 0.5704 | 105.96 | 3512 |
| `a3` | DISTILBERT | SST2 | **69.72% ± 0.00%** | `[69.7, 69.7]` | 1.0000 | 0.7244 | 0.3971 | 58.70 | 2314 |
| `cur` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.0633 | 53.92 | 1968 |