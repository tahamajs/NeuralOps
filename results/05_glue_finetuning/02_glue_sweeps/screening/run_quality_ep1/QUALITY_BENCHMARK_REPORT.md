# Phase 2 Quality Benchmark & Statistical Significance Report
**Generated:** run_quality_ep1  
**Hardware Platform:** NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Statistical Protocol:** $n=3$ independent random seeds (`[42, 123, 777]`), 95% Confidence Intervals, Paired Student's t-test vs `full` baseline.  

## Quality & Statistical Significance Summary Table
| Architecture | Model | Dataset | Accuracy (%) | 95% CI | $p$-value vs Base | F1 Score | Fidelity | Fwd (ms) | VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full` | DISTILBERT | SST2 | **90.25% ± 0.00%** | `[90.2, 90.2]` | -- | 0.9059 | 1.0000 | 67.07 | 1847 |
| `nasc_attention` | DISTILBERT | SST2 | **81.88% ± 0.00%** | `[81.9, 81.9]` | 1.0000 | 0.8184 | 0.8140 | 232.26 | 2600 |
| `minimal_svd` | DISTILBERT | SST2 | **79.13% ± 0.00%** | `[79.1, 79.1]` | 1.0000 | 0.7807 | 0.7627 | 86.47 | 1848 |
| `leka_v3` | DISTILBERT | SST2 | **78.33% ± 0.00%** | `[78.3, 78.3]` | 1.0000 | 0.7800 | 0.7407 | 171.53 | 2469 |
| `svd_linear_attention` | DISTILBERT | SST2 | **75.11% ± 0.00%** | `[75.1, 75.1]` | 1.0000 | 0.7264 | 0.6680 | 101.38 | 1847 |
| `aegis_v6` | DISTILBERT | SST2 | **72.13% ± 0.00%** | `[72.1, 72.1]` | 1.0000 | 0.6563 | 0.6265 | 277.48 | 2991 |
| `griffin_ssd` | DISTILBERT | SST2 | **62.50% ± 0.00%** | `[62.5, 62.5]` | 1.0000 | 0.7159 | -0.0070 | 50.77 | 1382 |
| `adaptive_span` | DISTILBERT | SST2 | **58.26% ± 0.00%** | `[58.3, 58.3]` | 1.0000 | 0.7055 | -0.0115 | 37.90 | 1311 |
| `block_transformer_hiba` | DISTILBERT | SST2 | **53.10% ± 0.00%** | `[53.1, 53.1]` | 1.0000 | 0.6837 | 0.0609 | 33.06 | 1310 |
| `cosformer` | DISTILBERT | SST2 | **52.87% ± 0.00%** | `[52.9, 52.9]` | 1.0000 | 0.6826 | -0.0244 | 46.94 | 1312 |
| `mamba` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.0458 | 22.59 | 1240 |
| `gated_deltanet_block` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.1300 | 48.92 | 1405 |
| `spectron` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.1366 | 34.86 | 1077 |
| `performer_attention` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.1455 | 48.80 | 1320 |
| `linformer_attention` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.1417 | 39.04 | 1322 |
| `galerkin_attention` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.0923 | 45.37 | 1312 |
| `soa_attn` | DISTILBERT | SST2 | **50.92% ± 0.00%** | `[50.9, 50.9]` | 1.0000 | 0.6748 | 0.1870 | 32.64 | 1143 |
| `svd_mamba` | DISTILBERT | SST2 | **49.08% ± 0.00%** | `[49.1, 49.1]` | 1.0000 | 0.0000 | 1.0000 | 355.58 | 4744 |
| `samba_m3` | DISTILBERT | SST2 | **0.00% ± 0.00%** | `[0.0, 0.0]` | **0.0010** (p < 0.01) | 0.0000 | 1.0000 | 0.00 | 0 |
| `jamba2` | DISTILBERT | SST2 | **0.00% ± 0.00%** | `[0.0, 0.0]` | **0.0010** (p < 0.01) | 0.0000 | 1.0000 | 0.00 | 0 |