# Phase 9: Final Submission Readiness Checklist

**Date:** 2026-09-20  
**Status:** SUBMISSION READY  

This checklist certifies the complete resolution of all code-paper discrepancies, mathematical formulations, structural references, empirical data grounding, figure formatting, citation audits, and clean PDF compilation for the A3 research paper.

---

## 1. Compliance and Quality Checklist

- [x] **Phase 0: Complete Inventory** (`paper/_PAPER_CODE_INVENTORY.md`)
  - Absolute paths mapped for all paper `.tex` files, code modules (`a3_compressor.py`, `pruning_utils.py`, `run_all_4_a3.py`), and result files.
  - All 73 label/ref definitions audited.
  - Zero unresolved `??`, `TODO`, or `FIXME` occurrences in source text.

- [x] **Phase 1: Code-Paper Alignment** (`paper/_ALIGNMENT_FIXED.md`)
  - Discrepancy 1 ($\tau$): Promoted $\tau=0.50$ default while preserving $\tau=0.40$ baseline.
  - Discrepancy 2 (Rank): Reconciled $r=64$ default / candidate dimension vs $r=128$ sweep rank.
  - Discrepancy 3 (Calibration): Reconciled to $N=500$ sequences uniformly across text and tables.
  - Discrepancy 4 (Rank ceiling): Aligned to $r_{\mathrm{ceiling}} = \min(\lfloor 0.90 r_{\mathrm{break\_even}}\rfloor, 320)$.
  - Discrepancy 5 (LoRA instantiation): Clarified base linear layer behavior (`lora_rank = 0` on plain layers).
  - Discrepancy 6 (Rank escalation step): Stepped toward $r_{\mathrm{ceiling}}$ with floor step 16.
  - Discrepancy 7 (LayerNorm recalibration): Fixed ratio compensation $\gamma \odot \mathrm{clamp}(\frac{\sigma_{\mathrm{target}}}{\sigma_{\mathrm{current}}}, 0.5, 2.0)$.
  - Discrepancy 8 (Weight sensitivity scaling): Aligned scale vector $\sqrt{\mathrm{clamp}(F, \min=10^{-8}) + \epsilon}$ and RMS scale normalization.

- [x] **Phase 2: Mathematical Formulations** (`paper/_MATH_FIXED.md`)
  - Fixed Equations (4.5), (4.8), (4.9), (4.10a), (4.11), (4.13), and (3.7).
  - Reconciled proofs and bounds in Appendices A.3, A.4, A.7, and A.8.

- [x] **Phase 3: References, Duplicates, and Structure** (`paper/_REFS_FIXED.md`)
  - Populated Section 1.5 (Notation) and Section 2.6 (Comparison Table) with substantive narrative text.
  - Added Section 11.1 Author Contributions (`sec:author_contributions`) and resolved pointer in Appendix E.5.1.
  - Generated dedicated Figure 9.2 (`figures/accuracy_vs_compression.pdf`) resolving duplicate Figure 9.1.
  - Deduplicated Table 3/14 (Notation Summary) and Table 11/15 (Expected Results).
  - Added missing captions and labels to Tables in `12_reproducibility.tex`, `D_negative_results.tex`, and `E_ethics.tex`.

- [x] **Phase 4: Scientific Reporting and Rigor** (`paper/_SCIENCE_FIXED.md`)
  - Documented under-converged RoBERTa baseline (78.56%) and LoRA head adaptation dynamics (+4.70%).
  - Clarified AG News 25.30% majority-class collapse resulting from unadapted 4-way classification heads.
  - Traced multi-seed numbers in Section D.6 directly to `results/multiseed/{42,123,456}/a3.json` (77.50% $\pm$ 0.00%).
  - Clarified empirical activation energy proxy vs. true Fisher information matrix.
  - Labeled INT8 compound quantization and intermediate LoRA step trajectory as analytical estimates.
  - Corrected environmental compute reduction math in Appendix E.2 (12.5h vs 1.0h = 12.5x compute reduction).
  - Added explicit Quality Gate Pass formula: $\mathrm{Acc}_{\mathrm{target}} \ge \max(56.00\%, \mathrm{Acc}_{\mathrm{majority}} + 5.0\%)$.

- [x] **Phase 5: Empirical Experiments Grounding** (`paper/_EXPERIMENTS_ADDED.md`)
  - Verified all 34 empirical tables against physical result files in `results/a3_tests/`, `results/unified_a3/`, and `results/multiseed/`.
  - Zero fabricated or unsupported numeric claims.

- [x] **Phase 6: Figures and Tables Cleanliness** (`paper/_FIGURES_FIXED.md`)
  - All vector PDFs verified and visually inspected.
  - Replaced double-dollar table syntax (`$$\times$`) with standard inline math.

- [x] **Phase 7: Citations Audit** (`paper/_CITATION_AUDIT.md`)
  - All references verified against real published papers.
  - Purged phantom artifacts (`author2024figure`, `author2024figure_1`, `author2024figure_2`, `author2026supplemental`, `zhu2024spectrum`).

- [x] **Phase 8: Clean PDF Compilation** (`paper/_COMPILE_LOG.md`)
  - Compiled with `pdflatex` to `paper/PerModulePaper/a3/main_final.pdf` (42 pages, 719 KB, 0 errors).
  - Synced changes to `paper_full.tex`.

---

## 2. Final Deliverable Paths

- **Final Compiled Paper PDF:** `paper/PerModulePaper/a3/main_final.pdf`
- **Master Paper LaTeX Source:** `paper/PerModulePaper/a3/main.tex`
- **Consolidated Paper LaTeX Source:** `paper/PerModulePaper/a3/paper_full.tex`
- **Phase Reports:**
  - `paper/_PAPER_CODE_INVENTORY.md`
  - `paper/_ALIGNMENT_FIXED.md`
  - `paper/_MATH_FIXED.md`
  - `paper/_REFS_FIXED.md`
  - `paper/_SCIENCE_FIXED.md`
  - `paper/_EXPERIMENTS_ADDED.md`
  - `paper/_FIGURES_FIXED.md`
  - `paper/_CITATION_AUDIT.md`
  - `paper/_COMPILE_LOG.md`
  - `paper/_FINAL_SUBMISSION_READY.md`
