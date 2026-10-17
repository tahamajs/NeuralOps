# 📊 Mathematical Ablations, HNC-SVD Homotopy & RoBERTa GLUE Audits

**Category Filter:** `ablation` | **Total Verified Records:** 63

| Module                   | Category | Family              | Setup / Dataset    | Accuracy | Speedup X | Peak Vram Mb | Method                   | Ver (Rev)   | Git SHA  | Timestamp  |
|:------------------------|-:--------|-:-------------------|-:------------------|-:--------|-:---------|-:------------|-:------------------------|-:-----------|-:--------|-:---------:|
| hnc_svd                  | ablation | hnc-svd             | Standard           | 62.50    | -         | -            | Naive DirectSVD          | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | 72.07    | -         | -            | HNC-SVD (Ours)           | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | 62.70    | -         | -            | Naive DirectSVD          | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | 73.73    | -         | -            | HNC-SVD (Ours)           | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | 66.31    | -         | -            | Naive DirectSVD          | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | 74.61    | -         | -            | HNC-SVD (Ours)           | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | 71.58    | -         | -            | Naive DirectSVD          | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | 77.05    | -         | -            | HNC-SVD (Ours)           | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | Raw Truncated SVD        | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC Aligned SVD (Ours)   | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | Raw Truncated SVD        | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC Aligned SVD (Ours)   | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | Raw Truncated SVD        | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC Aligned SVD (Ours)   | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | Raw Truncated SVD        | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC Aligned SVD (Ours)   | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | Raw Truncated SVD        | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC Aligned SVD (Ours)   | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | 0.68      | -            | HNC-SVD                  | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC-SVD                  | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC-SVD                  | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC-SVD                  | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| hnc_svd                  | ablation | hnc-svd             | Standard           | -        | -         | -            | HNC-SVD                  | v1.3.7 (r1) | ba32ad26 | 2026-10-17 |
| roberta_compression      | ablation | glue-benchmark      | RoBERTa-base SST-2 | 52.06    | -         | 1017.90      | Naive DirectSVD          | v1.3.7 (r1) | 09e6fdd6 | 2026-10-16 |
| roberta_compression      | ablation | glue-benchmark      | RoBERTa-base SST-2 | 73.62    | -         | 1475.50      | HNC-SVD (Ours)           | v1.3.7 (r1) | 09e6fdd6 | 2026-10-16 |
| roberta_compression      | ablation | glue-benchmark      | RoBERTa-base SST-2 | 47.82    | -         | 1010.70      | Naive DirectSVD          | v1.3.7 (r1) | 09e6fdd6 | 2026-10-16 |
| roberta_compression      | ablation | glue-benchmark      | RoBERTa-base SST-2 | 71.67    | -         | 1464.00      | HNC-SVD (Ours)           | v1.3.7 (r1) | 09e6fdd6 | 2026-10-16 |
| cosformer_regime_0       | ablation | kd-6pillar-ablation | Standard           | -        | -         | 488.90       | Regime 0: Zero-Shot (alp | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_regime_1       | ablation | kd-6pillar-ablation | Standard           | -        | -         | 566.90       | Regime 1: Naive CE (50 s | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_regime_2       | ablation | kd-6pillar-ablation | Standard           | -        | -         | 766.90       | Regime 2: Forward KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_regime_3       | ablation | kd-6pillar-ablation | Standard           | -        | -         | 766.90       | Regime 3: Reverse KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_regime_4       | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1029.20      | Regime 4: Decoupled KD ( | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_regime_5       | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1029.20      | Regime 5: 6-Pillar Suite | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gla_attention_regime_0   | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1029.20      | Regime 0: Zero-Shot (alp | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gla_attention_regime_1   | ablation | kd-6pillar-ablation | Standard           | -        | -         | 551.60       | Regime 1: Naive CE (50 s | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gla_attention_regime_2   | ablation | kd-6pillar-ablation | Standard           | -        | -         | 751.60       | Regime 2: Forward KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gla_attention_regime_3   | ablation | kd-6pillar-ablation | Standard           | -        | -         | 751.60       | Regime 3: Reverse KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gla_attention_regime_4   | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1013.90      | Regime 4: Decoupled KD ( | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gla_attention_regime_5   | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1014.40      | Regime 5: 6-Pillar Suite | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gdn2_attention_regime_0  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1014.40      | Regime 0: Zero-Shot (alp | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gdn2_attention_regime_1  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 816.00       | Regime 1: Naive CE (50 s | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gdn2_attention_regime_2  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1016.00      | Regime 2: Forward KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gdn2_attention_regime_3  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1016.00      | Regime 3: Reverse KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gdn2_attention_regime_4  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1278.20      | Regime 4: Decoupled KD ( | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| gdn2_attention_regime_5  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1278.80      | Regime 5: 6-Pillar Suite | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| pgdn_attention_regime_0  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1278.80      | Regime 0: Zero-Shot (alp | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| pgdn_attention_regime_1  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 681.40       | Regime 1: Naive CE (50 s | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| pgdn_attention_regime_2  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 881.40       | Regime 2: Forward KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| pgdn_attention_regime_3  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 881.40       | Regime 3: Reverse KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| pgdn_attention_regime_4  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1143.70      | Regime 4: Decoupled KD ( | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| pgdn_attention_regime_5  | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1144.20      | Regime 5: 6-Pillar Suite | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| mamba_block_regime_0     | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1144.20      | Regime 0: Zero-Shot (alp | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| mamba_block_regime_1     | ablation | kd-6pillar-ablation | Standard           | -        | -         | 643.40       | Regime 1: Naive CE (50 s | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| mamba_block_regime_2     | ablation | kd-6pillar-ablation | Standard           | -        | -         | 843.40       | Regime 2: Forward KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| mamba_block_regime_3     | ablation | kd-6pillar-ablation | Standard           | -        | -         | 843.40       | Regime 3: Reverse KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| mamba_block_regime_4     | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1105.70      | Regime 4: Decoupled KD ( | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| mamba_block_regime_5     | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1105.70      | Regime 5: 6-Pillar Suite | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_aasvd_regime_0 | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1105.70      | Regime 0: Zero-Shot (alp | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_aasvd_regime_1 | ablation | kd-6pillar-ablation | Standard           | -        | -         | 572.90       | Regime 1: Naive CE (50 s | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_aasvd_regime_2 | ablation | kd-6pillar-ablation | Standard           | -        | -         | 772.90       | Regime 2: Forward KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_aasvd_regime_3 | ablation | kd-6pillar-ablation | Standard           | -        | -         | 772.90       | Regime 3: Reverse KL (50 | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_aasvd_regime_4 | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1035.10      | Regime 4: Decoupled KD ( | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
| cosformer_aasvd_regime_5 | ablation | kd-6pillar-ablation | Standard           | -        | -         | 1035.10      | Regime 5: 6-Pillar Suite | v1.3.7 (r1) | c8fd39c0 | 2026-10-14 |
