# AdaSVD Compressor Paper (v2) Fix Report

**Date:** 2026-09-21  
**Target Paper:** [`paper/adasvd_compressor_paper_v2.tex`](paper/adasvd_compressor_paper_v2.tex)  
**PDF Output:** [`paper/adasvd_compressor_paper_v2.pdf`](paper/adasvd_compressor_paper_v2.pdf)  
**Extracted Numbers File:** [`paper/_PAPER_NUMBERS.csv`](paper/_PAPER_NUMBERS.csv)  

---

## 1. Numbers Reconciliation

All numbers were strictly extracted from verified execution CSV logs. **Zero numbers were invented.**

| Metric | Old Paper | New Paper (from CSV) | Source CSV File | Status |
|--------|-----------|----------------------|-----------------|--------|
| **SST-2 Acc (Seed 42)** | 85.32% | **88.19%** | `summary_verify_adasvd_gating_seed42.csv` | **CORRECTED** |
| **SST-2 Acc (Seed 123)** | N/A | **87.39%** | `summary_verify_adasvd_gating_seed123.csv` | **ADDED** |
| **SST-2 Acc (Seed 456)** | N/A | **86.12%** | `summary_verify_adasvd_gating_seed456.csv` | **ADDED** |
| **SST-2 Acc (Mean ± Std)** | N/A | **87.23% ± 1.04%** | Computed across 3 seeds | **ADDED** |
| **SST-2 F1 Score** | 0.8672 | **0.8884** (Mean: 0.8807) | `summary_verify_adasvd_gating_seed42.csv` | **CORRECTED** |
| **SST-2 Recon Error** | 0.6147 | **0.1912** | `summary_verify_adasvd_gating_seed42.csv` | **CORRECTED** |
| **SST-2 Params After** | 57,613,058 | **57,660,674** | `summary_verify_adasvd_gating_seed42.csv` | **CORRECTED** |
| **SST-2 Trainable Params** | 5,424,386 | **5,472,002** | `summary_verify_adasvd_gating_seed42.csv` | **CORRECTED** |
| **SST-2 Compression Rate** | 1.16x | **1.1612x** (13.88% reduction) | `summary_verify_adasvd_gating_seed42.csv` | **CORRECTED** |
| **SST-2 Peak VRAM** | 370.3 MB | **422.86 MB** | `summary_verify_adasvd_gating_seed42.csv` | **CORRECTED** |
| **SST-2 Throughput** | 75897 tok/s | **77646 tok/s** (Mean: 117964) | `summary_verify_adasvd_gating_seed42.csv` | **CORRECTED** |
| **MNLI Acc (full)** | 61.00% | **71.10%** | `results/distilbert_mnli_full_mnli_sweep/summary.csv` | **CORRECTED** |
| **MNLI F1 Score** | 0.6064 | **0.7076** | `results/distilbert_mnli_full_mnli_sweep/summary.csv` | **CORRECTED** |
| **MNLI Recon Error** | 0.6147 | **0.6147** | `results/distilbert_mnli_full_mnli_sweep/summary.csv` | **VERIFIED** |
| **AG News Acc** | 87.40% | **TBD** | `results/distilbert_agnews_full_agnews_sweep/` (missing) | **MARKED TBD** |

---

## 2. Sections Added

- [x] **Abstract Rewritten:** Completely rewritten to be comprehensive: clear problem statement, description of activation-aware Cholesky whitening, gating-aware importance weighting via non-linear derivative scaling, mean-shift bias absorption, verified empirical numbers on SST-2 (88.19% vs AASVD 87.16%), and novelty statement.
- [x] **Gating-Aware D Subsection (Section 4.2):** Formalized with Taylor expansion $\Delta Z \approx f'(Y) \odot (\Delta W X^\top)$, derivation of the analytic GELU derivative $\operatorname{GELU}'(y) = \Phi(y) + y \phi(y)$, and the expectation matrix $D_{jj} = \mathbb{E}[(\operatorname{GELU}'(Y_j))^2]$.
- [x] **Mean-Shift Bias Absorption Subsection (Section 4.3):** Explains activation channel offsets, derivation of $b_{\text{new}} = b + W \mu_X^\top$, zero inference rank waste, and the empirical 6.4x drop in unmodeled subspace variance ($0.2520 \to 0.0392$).
- [x] **Table 3: Comparison with State-of-the-Art Baselines (Rank 128):**
  - Dense Baseline: 90.71%
  - SVD-LLM: 90.94% (1.00x, error 0.7275)
  - DRONE: 90.71% (1.00x)
  - FWSVD: 90.71% (1.00x)
  - AASVD: 87.16% (1.2042x, error 0.8162)
  - **AdaSVD (Ours): 88.19% (1.1612x, error 0.1912)**
- [x] **Table 4: Multi-Seed Robustness Evaluation (Seeds 42, 123, 456):**
  - Seed 42: 88.19% (F1: 0.8884)
  - Seed 123: 87.39% (F1: 0.8807)
  - Seed 456: 86.12% (F1: 0.8730)
  - **Mean ± Std: 87.23% ± 1.04%**
- [x] **Table 5: Ablation Study (Seed 42, Rank 128):**
  - Config A (Var(Y), ALS Off): 88.19%
  - Config B (Gating GELU', ALS Off): 88.19%
  - Config C (Gating GELU', ALS On): 88.19%
  - Old V2 (Var(Y), ALS On): 86.93% (+1.26% degradation recovered)

---

## 3. Sections Fixed

- [x] **Related Work Expanded:** Added dedicated discussions and citations for:
  - SVD-LLM (Wang et al., ICLR 2025)
  - ASVD (Yuan et al., arXiv 2312.05821)
  - FWSVD (Hsu et al., ICLR 2022)
  - DRONE (Chen et al., NeurIPS 2021)
  - AdaSVD original (Li et al., arXiv 2502.01403)
  - SparseGPT (Frantar & Alistarh, ICML 2023)
  - AWQ (Lin et al., MLSys 2024)
  - LoRA (Hu et al., ICLR 2022)
- [x] **Theorem 4 Corrected:** Replaced the false claim that row-stochastic attention matrices have an induced $L_2$ operator norm bounded by 1. Stated and proved the correct bound $\|A\|_2 \le \sqrt{N}$ via Riesz-Thorin interpolation and Hölder inequalities, and updated the cascaded activation norm bound accordingly.
- [x] **PPL Removed from Encoders:** Table C.1 (Appendix C) updated to remove Perplexity (PPL) for DistilBERT and RoBERTa, reporting it exclusively for autoregressive language decoders (GPT-2).
- [x] **Double-Blind Anonymization:**
  - Replaced all instances of `D:\svd_project` and `D:/svd_project/` with `the experimental environment`.
  - Replaced `Codes/framework/compression_engine/` and internal framework paths with generic implementations.
  - Anonymized author metadata, affiliations, and repository URLs.
- [x] **Appendix E Shrunk:** Condensed from 8 redundant subsections to a clean, focused 1-page section covering:
  - E.1 Environmental Impact and Datacenter Power Accounting
  - E.2 Democratization and Open Access
  - E.3 Ethical and Safety Considerations
  - E.4 Deployment Implications for Edge and Cloud Infrastructures
- [x] **Compilation Verified:** Compiled cleanly via `pdflatex` through 2 passes with 0 fatal errors.

---

## 4. Remaining Issues & Notes

1. **AG News Data:** The AG News sweep directory (`results/distilbert_agnews_full_agnews_sweep/`) does not yet have a finished `summary.csv`. Marked strictly as **TBD** in Table 6 to avoid fabricating any metrics.
2. **Naming Clarification:** Explicitly stated in the Abstract, Introduction, and Related Work that this paper builds upon and extends the original AdaSVD formulation (Li et al., arXiv 2502.01403) with activation whitening, gating-aware importance weighting, and mean-shift bias absorption.
