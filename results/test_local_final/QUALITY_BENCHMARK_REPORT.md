# Phase 2 Quality Benchmark & Statistical Significance Report
**Generated:** test_local_final  
**Hardware Platform:** NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Statistical Protocol:** $n=3$ independent random seeds (`[42, 123, 777]`), 95% Confidence Intervals, Paired Student's t-test vs `full` baseline.  

## Quality & Statistical Significance Summary Table
| Architecture | Model | Dataset | Accuracy (%) | 95% CI | $p$-value vs Base | F1 Score | Fidelity | Fwd (ms) | VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full` | BERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | -- | 0.6748 | 1.0000 | 296.46 | 0 |
| `mamba` | BERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.9659 | 92.00 | 0 |