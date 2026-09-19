# Abstract Numbers Summary (Verified from CSVs)

## 2.1 Baseline Numbers
| Dataset | Baseline Accuracy | Chance Threshold | Classes | Backbone |
| :--- | :--- | :--- | :--- | :--- |
| SST-2 | 91.06% | 50.0% | 2 | DistilBERT-base |
| MNLI | 81.80% | 33.3% | 3 | DistilBERT-base |
| AG News | 94.30% | 25.0% | 4 | DistilBERT-base |

## 2.2 a3 vs. kmeans Comparison
| Method | SST-2 Accuracy | MNLI Accuracy | AG News Accuracy | Compression Ratio | Reconstruction Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| full (baseline) | 91.06% | 81.80% | 94.30% | 1.00× | -- |
| a3 (SVD rank-truncation) | 90.71% | 72.20% | 89.30% | 1.05× | 0.377 |
| kmeans (Vector Quantization) | 90.37% | 81.40% | [PENDING] | 2.73× | 0.032 |

- **MNLI Accuracy Delta**: kmeans (81.40%) - a3 (72.20%) = **+9.20 points**
- **Reconstruction Error Ratio**: a3 (0.377) / kmeans (0.032) = **11.7× lower reconstruction error** for kmeans.

## 2.3 Number of Modules Studied
- **SST-2 Evaluated**: 243 modules
- **MNLI Evaluated**: 271 modules
- **AG News Evaluated**: 87 modules (sweep active / in progress)
- **Total Unique Modules**: 277 unique modules

## 2.4 Non-Expansion Enforcement
- **REJECTED_EXPANSION Count on MNLI**: 80 modules
- **Largest Expansion**: `jamba` adding +93,367,296 parameters (+139.45% growth over 66.96M base)
- **Mathematical Break-even Rank for $d=768$**: $r = 384$ ($2 \times d \times r = d^2$)
- **Enforced Practical Rank Ceiling for Non-Expansion with recovery adapters**: $r \le 320$ (guaranteeing $\le 256$ in standard configs)

## 2.5 Status Distribution Summary
| Status | SST-2 | MNLI | AG News |
| :--- | :--- | :--- | :--- |
| PASS | 182 | 120 | 50 |
| REJECTED_EXPANSION | 10 | 80 | 33 |
| SKIP_BERT_ARCHITECTURAL | 17 | 43 | 0 |
| PASS_LOW | 15 | 9 | 1 |
| FAIL | 12 | 15 | 0 |
