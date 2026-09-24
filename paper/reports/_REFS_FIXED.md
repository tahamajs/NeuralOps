# Phase 3: References, Duplicates, and Structure Fixed

**Date:** 2026-09-20  
**Status:** COMPLETE  

This report details structural improvements, broken reference resolutions, table deduplications, and narrative section completions in the A3 paper.

---

## 1. Structural Fixes Inventory

1. **Populated Section 1.5 (Notation):**
   - Added descriptive paragraph defining bold uppercase matrix notation ($W$), bold lowercase vectors ($x$), Hadamard element-wise product ($\odot$), diagonal operator ($\mathrm{diag}$), Frobenius norm ($\|\cdot\|_F$), and factor dimensions ($U, V$). Added cross-reference pointing to Appendix A.8 Table~\ref{tab:app_notation_summary}.

2. **Populated Section 2.6 (Comparison Table):**
   - Added introductory text in `02_related.tex` contextualizing Table~\ref{tab:prior_work_comparison} across matrix factorization, data-aware SVD, post-training pruning, activation-aware quantization, and knowledge distillation.

3. **Resolved External Section Reference (11.7 Author Contributions):**
   - Added `\subsection{Author Contributions}\label{sec:author_contributions}` to `11_conclusion.tex` detailing individual author contributions.
   - Updated reference in `E_ethics.tex` from broken un-numbered text `See 11.7.` to `See Section~\ref{sec:author_contributions}.`.

4. **Resolved Duplicate Figure 9.1 (`fig:comparison_methods`):**
   - Identified that `fig_9_1.pdf` was an exact duplicate of the decision flowchart in Figure 9.1 (`fig:decision_flowchart`).
   - Generated a clean, publication-ready vector figure `paper/PerModulePaper/a3/figures/accuracy_vs_compression.pdf` plotting Downstream Accuracy vs. Whole-Model Compression Ratio for ALBERT, DistilBERT, TinyBERT, FWSVD, DRONE, and A3 ($\tau=0.40$ and $\tau=0.50$).
   - Updated Figure 9.2 in `09_discussion.tex` to point to `figures/accuracy_vs_compression.pdf`.

5. **Deduplicated Redundant Tables:**
   - **Table 3 & Table 14 (Notation Summary):** Removed redundant Table 3 from `03_background.tex` and replaced with an explicit text reference pointing to the comprehensive Table~\ref{tab:app_notation_summary} in Appendix A.8.
   - **Table 11 & Table 17 (Expected Reproducibility Results):** Replaced duplicate tabular block in `12_reproducibility.tex` with text referencing Table~\ref{tab:app_repro_expected} in Appendix B.10.

6. **Added Missing Table Captions & Labels:**
   - Table in `12_reproducibility.tex` (Dependencies): Added caption and label `\label{tab:repro_dependencies}`.
   - Table in `D_negative_results.tex` (Multi-Seed): Added caption and label `\label{tab:app_multiseed_eval}`.
   - Table in `E_ethics.tex` (Environmental Compute): Added caption and label `\label{tab:app_co2_comparison}`.

---

## 2. Integrity Verification
All 73 labels in the paper now resolve without broken pointers, duplicate identifiers, or missing captions.
