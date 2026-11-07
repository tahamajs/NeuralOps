# Academic Monograph, Undergraduate (BSc) Thesis & Defense Documentation

This directory contains the authoritative academic publications, undergraduate (BSc) thesis manuscripts, presentation slide decks, and build specifications for **NeuralOps** (`SVD_linear_Attention`).

---

## 1. Compilation Protocol & Prerequisites

All academic manuscripts and presentation decks are authored in LaTeX and compiled via standard TeX Live distributions (`pdflatex`, `xelatex`, and `bibtex`).

### 1.1 Environment Requirements
- **TeX Distribution**: TeX Live 2024+ or MacTeX 2024+
- **Compilers**: `pdflatex` (arXiv monograph and conference papers), `xelatex` (bilingual Persian/English undergraduate thesis with XePersian)
- **Bibliography Engines**: `bibtex` / `biber`
- **Recommended Font Packages**: TeX Gyre Termes, Libertine, Fira Code, IranNastaliq / Vazirmatn / X Nazanin (for Persian thesis)

### 1.2 Monograph Build Procedure (`pdflatex`)
To compile the comprehensive arXiv monograph:
```bash
cd academic/paper/arxiv
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### 1.3 Undergraduate (BSc) Thesis Build Procedure (`xelatex`)
To compile the university undergraduate thesis:
```bash
cd academic/thesis
xelatex -interaction=nonstopmode 810101504_main.tex
bibtex 810101504_main
xelatex -interaction=nonstopmode 810101504_main.tex
xelatex -interaction=nonstopmode 810101504_main.tex
```

---

## 2. Deliverables & Output PDF Inventory

All compiled PDF documents residing in the repository checkout are catalogued below (measured via `ls -l`):

### 2.1 Monograph & Conference Manuscripts
| Document | Relative Path | Measured Size | Status & Notes |
| :--- | :--- | :---: | :--- |
| **Comprehensive Monograph (arXiv)** | `academic/paper/arxiv/main.pdf` | 10.3 MB | Production build (1,418 pages, 0 undefined refs) |
| **NeurIPS 2026 Manuscript** | `academic/paper/neurips2026-submission/neurips2026_conference.pdf` | 64.4 MB | Draft — deadline missed (unsubmitted preprint) |
| **ICLR 2027 Manuscript** | `academic/paper/iclr2027-submission/iclr2027_conference.pdf` | 64.4 MB | Draft — deadline missed (anonymous double-blind) |
| **MLSys 2027 Manuscript** | `academic/paper/mlsys/mlsys2027_conference.pdf` | 68.3 MB | Draft — CFP status unverified |
| **ICML 2027 Manuscript** | `academic/paper/icml/icml2027_conference.pdf` | 64.3 MB | Draft — CFP status unverified |

### 2.2 Undergraduate (BSc) Thesis & University Defense
| Document | Relative Path | Measured Size | Focus & Scope |
| :--- | :--- | :---: | :--- |
| **BSc Thesis Main (810101504)** | `academic/thesis/810101504_main.pdf` | 205.4 MB | Authoritative university monograph (593 pages) |
| **BSc Defense Slides (Persian)** | `academic/thesis/810101504_defense.pdf` | 205.8 MB | Oral defense slide compilation (863 pages) |
| **BSc Defense Presentation (English)** | `academic/slides/master_defense_2026_en.pdf` | 43.6 MB | 50-frame English presentation deck |
| **BSc Defense Presentation (Bilingual)** | `academic/slides/master_defense_2026.pdf` | 43.7 MB | 50-frame Persian/English defense deck |
| **Summary Slide Deck** | `academic/slides/defense_presentation.pdf` | 99 KB | 10-frame executive overview deck |
| **University Proposal Document** | `academic/proposal/فرم پروپوزال کارشناسی.pdf` | ~450 KB | Signed undergraduate thesis proposal |

---

## 3. Hygiene & Build Artifact Exclusion

All generated build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`, `*.toc`, `*.nav`, `*.snm`, `*.vrb`, `*.pdf`) within `academic/` and `paper/` are strictly excluded from git tracking via root `.gitignore` (`academic/`, `paper/`) to prevent repository bloat. Production deliverables are permanently archived in the dedicated GitHub Academic Monographs Release Hub.
