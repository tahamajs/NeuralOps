# Academic Monograph, Master's Thesis & Defense Documentation

This directory contains the authoritative academic publications, master's thesis manuscripts, presentation slide decks, and build specifications for **NeuralOps** (`SVD_linear_Attention`).

---

## 1. Compilation Protocol & Prerequisites

All academic manuscripts and presentation decks are authored in LaTeX and compiled via standard TeX Live distributions (`pdflatex`, `xelatex`, and `bibtex`).

### 1.1 Environment Requirements
- **TeX Distribution**: TeX Live 2024+ or MacTeX 2024+
- **Compilers**: `pdflatex` (arXiv monograph and conference papers), `xelatex` (bilingual Persian/English master's thesis with XePersian)
- **Bibliography Engines**: `bibtex` / `biber`
- **Recommended Font Packages**: TeX Gyre Termes, Libertine, Fira Code, IranNastaliq / Vazirmatn (for Persian thesis)

### 1.2 Monograph Build Procedure (`pdflatex`)
To compile the comprehensive arXiv monograph:
```bash
cd academic/paper/arxiv
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### 1.3 Master's Thesis Build Procedure (`xelatex`)
To compile the university master's thesis:
```bash
cd academic/thesis
xelatex -interaction=nonstopmode 810101504_main.tex
bibtex 810101504_main
xelatex -interaction=nonstopmode 810101504_main.tex
xelatex -interaction=nonstopmode 810101504_main.tex
```

---

## 2. Deliverables & Output PDF Inventory

All compiled PDF documents residing in the repository checkout are catalogued below.
*(Note: Page counts are marked as `unverified (pdfinfo not available)` because `pdfinfo` is not installed in the local CLI environment).*

### 2.1 Monograph & Preprints
| Document | Relative Path | Size | Page Count Verification |
| :--- | :--- | :--- | :--- |
| **Comprehensive Monograph (arXiv)** | `academic/paper/arxiv/main.pdf` | 10.8 MB | unverified (pdfinfo not available) |
| **NeurIPS 2026 Submission** | `academic/paper/neurips2026-submission/neurips2026_conference.pdf` | ~1.2 MB | unverified (pdfinfo not available) |
| **ICLR 2027 Submission** | `academic/paper/iclr2027-submission/iclr2027_conference.pdf` | ~1.1 MB | unverified (pdfinfo not available) |
| **MLSys 2027 Submission** | `academic/paper/mlsys/mlsys2027_conference.pdf` | ~1.0 MB | unverified (pdfinfo not available) |
| **ICML 2027 Submission** | `academic/paper/icml/icml2027_conference.pdf` | ~1.0 MB | unverified (pdfinfo not available) |

### 2.2 Master's Thesis & University Defense
| Document | Relative Path | Size | Page Count Verification |
| :--- | :--- | :--- | :--- |
| **Master's Thesis Main (810101504)** | `academic/thesis/810101504_main.pdf` | 215.4 MB | unverified (pdfinfo not available) |
| **Master's Defense Slides (Persian)** | `academic/thesis/810101504_defense.pdf` | 180.5 MB | unverified (pdfinfo not available) |
| **Master's Defense Presentation (English)** | `academic/slides/master_defense_2026_en.pdf` | 45.8 MB | unverified (pdfinfo not available) |
| **Master's Defense Presentation (Bilingual)** | `academic/slides/master_defense_2026.pdf` | 45.9 MB | unverified (pdfinfo not available) |
| **Summary Slide Deck** | `academic/slides/defense_presentation.pdf` | 102 KB | unverified (pdfinfo not available) |
| **University Proposal Document** | `academic/proposal/فرم پروپوزال کارشناسی.pdf` | ~500 KB | unverified (pdfinfo not available) |

---

## 3. Hygiene & Build Artifact Exclusion

All LaTeX compilation scratch files (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`, `*.toc`, `*.nav`, `*.snm`, `*.vrb`, `texput.log`) are strictly excluded via `.gitignore` to prevent repository bloating.
Only finalized, release-tagged deliverables are committed or hosted on the academic release hub.
