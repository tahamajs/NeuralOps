# Paper Directory Consolidation & Reorganization Log

**Date:** October 16, 2026  
**Auditor:** Principal Research Systems Architect, LaTeX Typesetting Engineer  
**Repository:** `NeuralOps` (`SVD_linear_Attention` v1.3.7)  

---

## 1. Executive Summary

In accordance with Phase 4 production hardening directives, fragmented, duplicate, and clutter files across the `paper/` directory were audited and consolidated into the standardized canonical layout.

---

## 2. Directory Relocations & Moves Performed

### A. Root Clutter Relocations
The following 11 intermediate screening files and caches at the root of `paper/` were cleanly relocated to `paper/audit/`:
1. `paper/_ALL_MODULES_ANALYSIS.csv` $\to$ `paper/audit/_ALL_MODULES_ANALYSIS.csv`
2. `paper/_ALL_MODULES_ANALYSIS.md` $\to$ `paper/audit/_ALL_MODULES_ANALYSIS.md`
3. `paper/_ALL_MODULES_VERIFIED.md` $\to$ `paper/audit/_ALL_MODULES_VERIFIED.md`
4. `paper/_ANOMALY_REPORT.md` $\to$ `paper/audit/_ANOMALY_REPORT.md`
5. `paper/_PIPELINE_QUEUE.json` $\to$ `paper/audit/_PIPELINE_QUEUE.json`
6. `paper/_PROGRESS.md` $\to$ `paper/audit/_PROGRESS.md`
7. `paper/_SPECTRA_V2_REPORT.md` $\to$ `paper/audit/_SPECTRA_V2_REPORT.md`
8. `paper/_VERIFICATION_LOG.md` $\to$ `paper/audit/_VERIFICATION_LOG.md`
9. `paper/_verification_results.json` $\to$ `paper/audit/_verification_results.json`
10. `paper/.paper_audit_results.json` $\to$ `paper/audit/.paper_audit_results.json`
11. `paper/.paper_list_cache.json` $\to$ `paper/audit/.paper_list_cache.json`

### B. Per-Module Monograph Unification
- Deleted deprecated `paper/PerModulePaper` symlink.
- Verified `paper/build_paper/` as the canonical storage directory containing all 247 per-module monograph packages.
- Created standardized symlink `paper/per_module -> build_paper` fulfilling the target layout.

### C. Script Reorganization
- Replicated `paper/build_paper.py` into `paper/scripts/build_paper.py` to consolidate paper generation tooling alongside `enhanced_sections.py` and `enrich_module_paper.py`.
- Preserved root `paper/build_paper.py` for backward CLI compatibility.

---

## 3. Final Production Layout

```
paper/
├── iclr2027-submission/               # Primary ICLR conference paper (29 pages, clean)
│   ├── iclr2027_conference.tex
│   ├── iclr2027_conference.bib
│   ├── tables/                        # 17 LaTeX table modules
│   ├── figures/                       # 5 high-resolution PNG figures
│   └── results/                       # Raw screening and benchmark CSVs
├── build_paper/                       # Canonical per-module monographs (247 directories)
├── per_module -> build_paper          # Target symlink alias
├── scripts/                           # Paper generation and audit tooling
│   ├── build_paper.py
│   ├── enhanced_sections.py
│   └── enrich_module_paper.py
├── figures/                           # Master vector graphics (PDF/SVG/PNG)
├── tables/                            # Supplementary tables
├── reports/                           # 66 Technical analysis monographs
├── audit/                             # Audit reports and screening data
└── README.md                          # Layout and reproduction documentation
```
