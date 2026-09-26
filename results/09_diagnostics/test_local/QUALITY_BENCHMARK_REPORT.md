# Phase 2 Quality Benchmark & Statistical Significance Report
**Generated:** test_local  
**Hardware Platform:** NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Statistical Protocol:** $n=3$ independent random seeds (`[42, 123, 777]`), 95% Confidence Intervals, Paired Student's t-test vs `full` baseline.  

## Quality & Statistical Significance Summary Table
| Architecture | Model | Dataset | Accuracy (%) | 95% CI | $p$-value vs Base | F1 Score | Fidelity | Fwd (ms) | VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full` | BERT | SST2 | **52.34% ± 0.00%** | `[52.3, 52.3]` | -- | 0.6872 | 1.0000 | 439.02 | 0 |
| `mamba` | BERT | SST2 | **0.00% ± 0.00%** | `[0.0, 0.0]` | 1.0000 | 0.0000 | 1.0000 | 0.00 | 0 |