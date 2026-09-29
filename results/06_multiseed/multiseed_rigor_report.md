# Multi-Seed Statistical Rigor Evaluation Report

- **Evaluation Dataset**: SST-2 validation set (N=872)
- **Seeds**: `[42, 123, 456]`
- **Recovery Fine-Tuning**: 50 steps (AdamW, FP16)
- **Generated Timestamp**: `2026-09-28 14:34:43`

## Summary Table (Mean ± Std Across Deterministic Seeds)

| Model | Module | Category | Completed Seeds | Accuracy (%) | F1 Score | Compression | Recon Error | Peak VRAM | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `distilbert` | `adasvd_compressor` | compressor | 2/3 | 87.73% | 0.8834 | 1.21x | 0.3560 | 493.1 MB | `PASS` |
| `distilbert` | `a3_compressor` | compressor | 2/3 | 90.25% | 0.9069 | 1.03x | 0.3493 | 503.3 MB | `PASS` |
| `distilbert` | `kronecker_compressor` | compressor | 2/3 | 90.71% | 0.9109 | 1.04x | — | 478.4 MB | `PASS` |
| `distilbert` | `mlorq_compressor` | compressor | 2/3 | 90.83% | 0.9109 | 1.00x | 0.0290 | 510.4 MB | `PASS` |
| `distilbert` | `rpca_compressor` | compressor | 2/3 | 90.77% | 0.9108 | 1.00x | — | 457.0 MB | `PASS` |
| `distilbert` | `direct_svd` | compressor | 2/3 | 90.60% | 0.9099 | 1.00x | 0.5882 | 446.0 MB | `PASS` |
| `distilbert` | `saes_svd_v2` | compressor | 2/3 | 90.71% | 0.9094 | 1.00x | — | 457.0 MB | `PASS` |
| `distilbert` | `salr` | compressor | 2/3 | 90.77% | 0.9108 | 1.00x | — | 457.0 MB | `PASS` |
| `distilbert` | `oats_compressor` | compressor | 2/3 | 90.82% | 0.9114 | 1.00x | 0.4116 | 472.1 MB | `PASS` |
| `distilbert` | `cure` | compressor | 2/3 | 90.60% | 0.9099 | 1.00x | — | 446.0 MB | `PASS` |
| `distilbert` | `admm_compressor` | compressor | 2/3 | 90.37% | 0.9060 | 1.00x | 0.1564 | 457.0 MB | `PASS` |
| `roberta` | `adasvd_compressor` | compressor | 2/3 | 83.26% | 0.8463 | 1.23x | 0.3276 | 503.2 MB | `PASS` |
| `roberta` | `a3_compressor` | compressor | 2/3 | 93.06% | 0.9318 | 1.02x | 0.3449 | 493.7 MB | `PASS` |
| `roberta` | `kronecker_compressor` | compressor | 2/3 | 91.57% | 0.9181 | 1.04x | — | 504.0 MB | `PASS` |
| `roberta` | `mlorq_compressor` | compressor | 2/3 | 94.44% | 0.9458 | 1.00x | 0.0402 | 518.1 MB | `PASS` |
| `roberta` | `rpca_compressor` | compressor | 2/3 | 94.16% | 0.9429 | 1.00x | — | 457.1 MB | `PASS` |
| `roberta` | `direct_svd` | compressor | 2/3 | 93.23% | 0.9353 | 1.00x | 0.6008 | 457.1 MB | `PASS` |
| `roberta` | `saes_svd_v2` | compressor | 2/3 | 93.98% | 0.9411 | 1.00x | — | 457.1 MB | `PASS` |
| `roberta` | `salr` | compressor | 2/3 | 93.92% | 0.9405 | 1.00x | — | 457.1 MB | `PASS` |
| `roberta` | `oats_compressor` | compressor | 2/3 | 93.87% | 0.9404 | 1.00x | 0.2050 | 514.6 MB | `PASS` |
| `roberta` | `cure` | compressor | 2/3 | 93.23% | 0.9353 | 1.00x | — | 457.1 MB | `PASS` |
| `roberta` | `admm_compressor` | compressor | 2/3 | 93.18% | 0.9318 | 1.00x | 0.1663 | 457.1 MB | `PASS` |
