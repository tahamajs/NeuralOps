# Comprehensive Forensic Audit: `paper/` Directory Structure Map

**Author:** Principal Research Systems Architect, LaTeX Typesetting Engineer, and Repository Structure Auditor  
**Date:** October 16, 2026  
**Repository:** `NeuralOps` (`SVD_linear_Attention` v1.3.7)  
**Commit Baseline:** `d5bffe071a`  

---

## 1. Executive Summary

A comprehensive recursive forensic audit of the `paper/` directory was conducted. The directory serves as the dual nexus of the project:
1. The **Primary Academic Submission**: The publication-ready ICLR 2027 conference submission manuscript located in `paper/iclr2027-submission/`.
2. The **Per-Module Monograph Library**: 247 auto-generated, peer-review format research monographs across all canonical and variant modules located in `paper/build_paper/`.
3. Supporting research infrastructure: figures, experimental tables, compilation templates, and audit reports.

---

## 2. Recursive Structural ASCII Map

```
paper/
├── README.md                                   # Root documentation explaining layout & build protocols
├── build_paper.py                              # Master generator script for all 247 per-module monographs
├── PerModulePaper -> build_paper               # Symlink alias for per-module monographs
│
├── iclr2027-submission/                        # PRIMARY SUBMISSION (ICLR 2027)
│   ├── iclr2027_conference.tex                 # Master LaTeX manuscript (26 pages, clean)
│   ├── iclr2027_conference.bib                 # Master BibTeX database (87 verified entries)
│   ├── iclr2027_conference.sty                 # Official ICLR 2027 style package
│   ├── iclr2027_conference.bst                 # BibTeX formatting style
│   ├── math_commands.tex                       # Standard deep learning math operators
│   ├── natbib.sty, fancyhdr.sty                # Typesetting dependencies
│   ├── iclr2027_conference.pdf                 # Compiled production PDF
│   ├── figures/                                # Core submission figures
│   │   ├── pareto_accuracy_vs_compression.png
│   │   ├── pareto_accuracy_vs_throughput.png
│   │   ├── pareto_accuracy_vs_vram.png
│   │   ├── pareto_frontier.png
│   │   └── pareto_ppl_vs_compression.png
│   ├── tables/                                 # 16 Canonical LaTeX table files
│   │   ├── table1_cross_arch.tex
│   │   ├── table3_baselines.tex
│   │   ├── table4_multiseed.tex
│   │   ├── table5_all_modules.tex
│   │   ├── table6_decoder.tex
│   │   ├── table7_long_context.tex
│   │   ├── table8_pareto.tex
│   │   ├── table_ablation.tex
│   │   ├── table_all_131_attentions.tex
│   │   ├── table_all_72_compressors.tex
│   │   ├── table_kd_regimes.tex
│   │   ├── table_lora_rank.tex
│   │   ├── table_roberta_hnc.tex
│   │   ├── table_triton_kernels.tex
│   │   ├── table_13_flagship_triton_kernels.tex
│   │   └── table_18_native_cuda_kernels_3level.tex
│   └── results/                                # Raw benchmarking CSV and JSON outputs
│       ├── benchmark_summary_v2.csv
│       ├── master_aggregated_results.csv
│       ├── mnli_summary.csv
│       ├── sst2_summary.csv
│       └── ...
│
├── build_paper/                                # AUTO-GENERATED MONOGRAPHS (247 Modules)
│   ├── a3_compressor/                          # Per-module monograph package
│   │   ├── 00_meta.md to E_ethics.md           # 19 standalone structured sections
│   │   ├── 00_meta.tex to E_ethics.tex         # LaTeX counterparts
│   │   ├── paper_full.md                       # Full concatenated Markdown monograph
│   │   ├── paper_full.tex                      # Full concatenated LaTeX monograph
│   │   └── main.tex                            # Standalone compilable document
│   ├── adasvd_compressor/
│   ├── mamba_block/
│   └── [244 additional module subdirectories]
│
├── figures/                                    # High-resolution vector assets (PDF / SVG / PNG)
│   ├── ablation_6pillar_recovery.{pdf,png}
│   ├── fig_niah_passkey_heatmap.{pdf,png,svg}
│   ├── figure_fla_scaling.{pdf,png,svg}
│   ├── figure_pareto_wikitext2.{pdf,png,svg}
│   ├── figure_recovery_ablation_6pillar.{pdf,png,svg}
│   ├── pareto_frontier_wikitext2.{pdf,png}
│   └── roberta_top15_attentions.{pdf,png}
│
├── tables/                                     # Standalone and ablation tables
│   ├── table_fla_scaling.tex
│   ├── table_recovery_ablation_6pillar.tex
│   └── table_wikitext2_pareto.tex
│
├── scripts/                                    # Monograph generation utilities
│   ├── __init__.py
│   ├── enhanced_sections.py
│   └── enrich_module_paper.py
│
├── reports/                                    # 66 Technical analysis reports & templates
│   ├── UNIVERSAL_PAPER_TEMPLATE.md
│   ├── FULL_PAPER_TEMPLATE.md
│   ├── adasvd_compressor_paper_v2.tex
│   ├── REFERENCES.bib
│   └── ...
│
├── audit/                                      # Monograph build verification reports
│   ├── _AUDIT_SUMMARY.md
│   ├── _CITATION_AUDIT.md
│   └── _DEEP_VERIFICATION.md
│
├── iclr2027-abstract/                          # Early abstract preparation workspace
│   ├── abstract.tex, abstract.txt, abstract_numbers.md
│   └── results/
│
├── templates/ & template/                      # LaTeX formatting templates
│   └── _TEMPLATE/, _shared/
│
├── papers/                                     # Legacy experimental exploratory drafts
│   ├── _baseline_power.json, _optimized_power.json
│   ├── _rank_ablation/
│   └── modules/
│
└── _archive/                                   # Deprecated / superseded prototype modules
    ├── nli_softmax/
    ├── pruning_utils/
    ├── ptp/
    ├── replaceme_compressor/
    ├── minimal_prism/
    ├── minimal_aurora/
    ├── auxiliary/
    └── squeezing_heads_distillation_layer/
```

---

## 3. Structural Classification Table

| Directory / Artifact | Classification Category | File Count | Primary Function | Retention Recommendation |
|:---|:---|:---:|:---|:---|
| `paper/iclr2027-submission/` | Primary Submission | 26 | Main ICLR 2027 conference paper, bib, tables, figures, styles | **KEEP CANONICAL** |
| `paper/build_paper/` | Auto-Generated Monographs | 247 folders (~10,000 files) | Complete 19-section research papers for each module | **KEEP CANONICAL** |
| `paper/PerModulePaper` | Symlink Alias | 1 | Symlink alias pointing to `build_paper/` | **CONSOLIDATE / RETAIN** |
| `paper/figures/` | High-Res Vector Assets | 18 | Master vector PDF/SVG figures for camera-ready inclusion | **KEEP** |
| `paper/tables/` | Supplementary Tables | 3 | Standalone micro-benchmark tables | **KEEP** |
| `paper/scripts/` | Tooling & Scripts | 3 | Module paper enrichment and section builders | **MOVE / ORGANIZE** |
| `paper/reports/` | Technical Monographs | 66 | Subsystem reports, empirical logs, template specs | **KEEP** |
| `paper/audit/` | Audit Logs | 3 | Verification logs for monograph generation | **KEEP** |
| `paper/iclr2027-abstract/` | Drafts & Abstract | 8 | Pre-submission abstract drafts | **ARCHIVE / KEEP** |
| `paper/templates/` | Templates | 5 | Reusable scaffolding templates | **KEEP** |
| `paper/papers/` | Exploratory Workspace | 5 | Early power and rank sweep JSONs | **ARCHIVE** |
| `paper/_archive/` | Deprecated Archives | 8 folders | Deprecated prototypes superseded by canonical registry | **KEEP ARCHIVED** |
| `paper/_ALL_MODULES_*.{csv,md}` | Root Clutter Files | 11 files | Intermediate screening results generated at root of `paper/` | **MOVE TO `paper/audit/`** |

---

## 4. Symlinks and Content Duplication Analysis

1. **Symlink `paper/PerModulePaper`**:
   - `paper/PerModulePaper -> build_paper`
   - **Resolution**: Target `paper/build_paper` exists and contains all 247 module folders. Resolved valid.
2. **Duplicate Content Check**:
   - `paper/iclr2027-abstract/` contains copies of `iclr2027_conference.sty`, `math_commands.tex`, `natbib.sty`, `fancyhdr.sty`. These are local copies ensuring standalone compilation of the abstract booklet.
   - `paper/build_paper/` vs `paper/PerModulePaper/`: Not duplicate storage, since `PerModulePaper` is an explicit symlink to `build_paper`.

---

## 5. Orphaned & Stale File Remediation Plan

The following 11 clutter files currently sitting at the root of `paper/` will be cleanly relocated to `paper/audit/` during Phase 4:
- `_ALL_MODULES_ANALYSIS.csv` $\to$ `paper/audit/`
- `_ALL_MODULES_ANALYSIS.md` $\to$ `paper/audit/`
- `_ALL_MODULES_VERIFIED.md` $\to$ `paper/audit/`
- `_ANOMALY_REPORT.md` $\to$ `paper/audit/`
- `_PIPELINE_QUEUE.json` $\to$ `paper/audit/`
- `_PROGRESS.md` $\to$ `paper/audit/`
- `_SPECTRA_V2_REPORT.md` $\to$ `paper/audit/`
- `_VERIFICATION_LOG.md` $\to$ `paper/audit/`
- `_verification_results.json` $\to$ `paper/audit/`
- `.paper_audit_results.json` $\to$ `paper/audit/`
- `.paper_list_cache.json` $\to$ `paper/audit/`
