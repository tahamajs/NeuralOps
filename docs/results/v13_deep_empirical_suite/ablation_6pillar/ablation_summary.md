# 6-Pillar Accuracy-Recovery & Distillation Empirical Ablation Suite
**Hardware**: NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Task**: Pure Student Subquadratic Surgery Recovery (`alpha=1.0`, `window_size=0`) across 6 Architectures & 6 Training Regimes (50 steps each).

| Architecture | Training Regime | Cosine Sim to Teacher (↑) | KL Div to Teacher (↓) | Peak VRAM (MB) | Time (s) |
|---|---|:---:|:---:|:---:|:---:|
| **Cosformer** | Regime 0: Zero-Shot (alpha=1.0) | 0.5519 | 2.9356 | 488.9 | 0.00s |
| **Cosformer** | Regime 1: Naive CE (50 steps) | 0.2286 | 4.4928 | 566.9 | 0.82s |
| **Cosformer** | Regime 2: Forward KL (50 steps) | 0.9495 | 0.3248 | 766.9 | 1.15s |
| **Cosformer** | Regime 3: Reverse KL (50 steps) | 0.9491 | 0.3271 | 766.9 | 1.41s |
| **Cosformer** | Regime 4: Decoupled KD (50 steps) | 0.9487 | 0.3302 | 1029.2 | 1.56s |
| **Cosformer** | Regime 5: 6-Pillar Suite (50 steps) | 0.9184 | 0.5125 | 1029.2 | 2.09s |
| **Chunk GLA** | Regime 0: Zero-Shot (alpha=1.0) | 0.0994 | 5.8951 | 1029.2 | 0.00s |
| **Chunk GLA** | Regime 1: Naive CE (50 steps) | 0.1578 | 4.8950 | 551.6 | 2.54s |
| **Chunk GLA** | Regime 2: Forward KL (50 steps) | 0.8293 | 1.0196 | 751.6 | 0.97s |
| **Chunk GLA** | Regime 3: Reverse KL (50 steps) | 0.8197 | 1.0678 | 751.6 | 1.27s |
| **Chunk GLA** | Regime 4: Decoupled KD (50 steps) | 0.8265 | 1.0344 | 1013.9 | 1.50s |
| **Chunk GLA** | Regime 5: 6-Pillar Suite (50 steps) | 0.7871 | 1.2447 | 1014.4 | 1.97s |
| **GDN-2** | Regime 0: Zero-Shot (alpha=1.0) | 0.3793 | 4.0607 | 1014.4 | 0.00s |
| **GDN-2** | Regime 1: Naive CE (50 steps) | 0.2005 | 4.7151 | 816.0 | 9.75s |
| **GDN-2** | Regime 2: Forward KL (50 steps) | 0.9577 | 0.2723 | 1016.0 | 9.30s |
| **GDN-2** | Regime 3: Reverse KL (50 steps) | 0.9581 | 0.2691 | 1016.0 | 9.66s |
| **GDN-2** | Regime 4: Decoupled KD (50 steps) | 0.9563 | 0.2810 | 1278.2 | 9.85s |
| **GDN-2** | Regime 5: 6-Pillar Suite (50 steps) | 0.9514 | 0.3111 | 1278.8 | 11.34s |
| **PGDN** | Regime 0: Zero-Shot (alpha=1.0) | 0.6808 | 2.0872 | 1278.8 | 0.00s |
| **PGDN** | Regime 1: Naive CE (50 steps) | 0.2129 | 4.6734 | 681.4 | 5.17s |
| **PGDN** | Regime 2: Forward KL (50 steps) | 0.9851 | 0.0981 | 881.4 | 5.22s |
| **PGDN** | Regime 3: Reverse KL (50 steps) | 0.9848 | 0.0998 | 881.4 | 5.93s |
| **PGDN** | Regime 4: Decoupled KD (50 steps) | 0.9850 | 0.0989 | 1143.7 | 6.18s |
| **PGDN** | Regime 5: 6-Pillar Suite (50 steps) | 0.9781 | 0.1423 | 1144.2 | 6.06s |
| **Mamba Block** | Regime 0: Zero-Shot (alpha=1.0) | 0.8416 | 1.0345 | 1144.2 | 0.00s |
| **Mamba Block** | Regime 1: Naive CE (50 steps) | 0.2223 | 4.5791 | 643.4 | 2.41s |
| **Mamba Block** | Regime 2: Forward KL (50 steps) | 0.9893 | 0.0710 | 843.4 | 2.61s |
| **Mamba Block** | Regime 3: Reverse KL (50 steps) | 0.9896 | 0.0694 | 843.4 | 2.90s |
| **Mamba Block** | Regime 4: Decoupled KD (50 steps) | 0.9896 | 0.0692 | 1105.7 | 3.19s |
| **Mamba Block** | Regime 5: 6-Pillar Suite (50 steps) | 0.9827 | 0.1124 | 1105.7 | 3.77s |
| **Cosformer + AASVD** | Regime 0: Zero-Shot (alpha=1.0) | 0.5713 | 2.8069 | 1105.7 | 0.00s |
| **Cosformer + AASVD** | Regime 1: Naive CE (50 steps) | 0.2258 | 4.5560 | 572.9 | 0.80s |
| **Cosformer + AASVD** | Regime 2: Forward KL (50 steps) | 0.9691 | 0.2022 | 772.9 | 1.03s |
| **Cosformer + AASVD** | Regime 3: Reverse KL (50 steps) | 0.9691 | 0.2014 | 772.9 | 1.32s |
| **Cosformer + AASVD** | Regime 4: Decoupled KD (50 steps) | 0.9694 | 0.1999 | 1035.1 | 1.55s |
| **Cosformer + AASVD** | Regime 5: 6-Pillar Suite (50 steps) | 0.9432 | 0.3624 | 1035.1 | 2.08s |
