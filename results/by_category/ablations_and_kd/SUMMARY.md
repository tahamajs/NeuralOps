# 📊 Mathematical Ablations, HNC-SVD Homotopy & RoBERTa GLUE Audits

**Category Filter:** `ablation` | **Total Verified Records:** 27

| Module              | Category | Family         | Setup / Dataset    | Accuracy | Speedup X | Peak Vram Mb | Method                 | Ver (Rev)   | Git SHA  | Timestamp  |
|:-------------------|-:--------|-:--------------|-:------------------|-:--------|-:---------|-:------------|-:----------------------|-:-----------|-:--------|-:---------:|
| hnc_svd             | ablation | hnc-svd        | Standard           | 62.50    | -         | -            | Naive DirectSVD        | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | 72.07    | -         | -            | HNC-SVD (Ours)         | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | 62.70    | -         | -            | Naive DirectSVD        | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | 73.73    | -         | -            | HNC-SVD (Ours)         | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | 66.31    | -         | -            | Naive DirectSVD        | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | 74.61    | -         | -            | HNC-SVD (Ours)         | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | 71.58    | -         | -            | Naive DirectSVD        | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | 77.05    | -         | -            | HNC-SVD (Ours)         | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | Raw Truncated SVD      | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC Aligned SVD (Ours) | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | Raw Truncated SVD      | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC Aligned SVD (Ours) | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | Raw Truncated SVD      | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC Aligned SVD (Ours) | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | Raw Truncated SVD      | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC Aligned SVD (Ours) | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | Raw Truncated SVD      | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC Aligned SVD (Ours) | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | 0.68      | -            | HNC-SVD                | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC-SVD                | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC-SVD                | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC-SVD                | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| hnc_svd             | ablation | hnc-svd        | Standard           | -        | -         | -            | HNC-SVD                | v1.3.7 (r1) | e65ad0b5 | 2026-10-16 |
| roberta_compression | ablation | glue-benchmark | RoBERTa-base SST-2 | 52.06    | -         | 1017.90      | Naive DirectSVD        | v1.3.7 (r1) | 09e6fdd6 | 2026-10-16 |
| roberta_compression | ablation | glue-benchmark | RoBERTa-base SST-2 | 73.62    | -         | 1475.50      | HNC-SVD (Ours)         | v1.3.7 (r1) | 09e6fdd6 | 2026-10-16 |
| roberta_compression | ablation | glue-benchmark | RoBERTa-base SST-2 | 47.82    | -         | 1010.70      | Naive DirectSVD        | v1.3.7 (r1) | 09e6fdd6 | 2026-10-16 |
| roberta_compression | ablation | glue-benchmark | RoBERTa-base SST-2 | 71.67    | -         | 1464.00      | HNC-SVD (Ours)         | v1.3.7 (r1) | 09e6fdd6 | 2026-10-16 |
