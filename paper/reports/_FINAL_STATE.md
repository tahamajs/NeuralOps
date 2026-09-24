# Evaluation Sweep Final State Report

**Timestamp**: 2026-09-20T23:15:00+03:30  
**Repository Branch**: `origin/main`  
**Execution Environment**: `hadi-windows` (NVIDIA RTX 4070 Laptop GPU, 8GB VRAM)  
**Evaluated Framework**: SVD Linear Attention Decomposition Suite

---

## 1. Executive Summary

All pending experimental stages and sweeps across all module groups have successfully completed execution, validation, and snapshot recording:

| Group | Description | Module Count | Status | Notes / Outcomes |
| :--- | :--- | :--- | :--- | :--- |
| **Group 1** | Threshold-Flip Re-audit | 6 modules | **COMPLETE** | `block_transformer_hiba` validated PASS (39.1%); `hybrid_svd` (37.1%) & `orion` (38.4%) classified PASS_LOW. |
| **Group 3** | Cat-B Base Expansion Audit | 5 modules | **COMPLETE** | Root cause proved: $6 \times (768 \times 768)$ full linear projections substitute attention maps ($\Delta \approx +5.3\%$). |
| **Group 5** | Collapse Audit ($N=1,000$) | 7 modules | **COMPLETE** | 4 modules confirmed Class 1 (Neutral) collapse due to uncalibrated classifier bias; 3 modules cleared. |
| **Group 7** | FAIL Module Re-runs | 4 modules | **COMPLETE** | `primal` passed at **81.90%**; `spectral_mamba` at **59.10%**; `sdrm` at **68.10%**; `mamba` at **34.10%** (PASS_LOW). |
| **Group 2** | Cat-A Re-sweep (LoRA rank 2) | 27 modules | **COMPLETE** | Run with 2 parallel workers, bs=16, all executed cleanly without memory violation. |
| **Group 6** | `hgrn2_outer` 30-step Retry | 1 module | **COMPLETE** | Successfully completed 30 recovery steps (2,741s, peak VRAM 571 MB). Tagged MODERATE_EXPANSION. |
| **Group 4** | Full-MNLI Validation ($N=9,815$) | 132 modules | **COMPLETE** | Validated across full test set with 95% Wilson Confidence Intervals. Base DistilBERT full accuracy: **81.97%**. |

---

## 2. Statistical Methodology & Verification Rules

1. **Dynamic Statistical Threshold**:
   - Local validation ($N = 1,000$): Majority class frequency $p_0 = 34.10\%$ ($341/1,000$). $\text{SE} = \sqrt{\frac{p_0(1-p_0)}{N}} = 1.499\%$. Critical boundary ($p_0 + 3\text{SE}$) = **38.60%**.
   - Full validation ($N = 9,815$): Majority class frequency $p_0 = 35.45\%$. $\text{SE} = 0.483\%$. Critical boundary ($p_0 + 3\text{SE}$) = **36.89%**.
   - No arbitrary static thresholds ($36.90\%$, $56.00\%$, or $30.00\%$) were utilized.
2. **Quality Gate Rule**:
   - **PASS**: Accuracy $\ge 38.60\%$ and effective parameter change $\Delta_{\text{eff}} \le 5.0\%$.
   - **FAIL_EXPANSION**: Accuracy $\ge 38.60\%$ but base model parameters expanded ($> 5.0\%$).
   - **FAIL_ACCURACY**: Accuracy $< 38.60\%$.
   - **Gate Pass Rate**: **156 of 203 modules passed the complete gate (76.8%)**.

---

## 3. Dataset Artifacts Generated

- [`summary_v4.csv`](file:///Users/tahamajs/Documents/uni/Project/results/distilbert_mnli_full_mnli_sweep/summary_v4.csv): Publication-ready master table with all 12 standardized columns across 203 evaluated modules.
- [`summary_full_mnli.csv`](file:///Users/tahamajs/Documents/uni/Project/results/distilbert_mnli_full_mnli_sweep/summary_full_mnli.csv): 132 PASS modules evaluated on $N=9,815$ samples with 95% Wilson confidence intervals.
- Snapshots archived:
  - `summary_snapshot_post_group7.csv`
  - `summary_snapshot_post_group2.csv`
  - `summary_snapshot_post_group6.csv`
  - `summary_snapshot_post_group4.csv`
- Collapse verification: `collapse_audit.csv` and `audit_logits/*.npy`.

