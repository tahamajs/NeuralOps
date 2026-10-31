# NeuralOps Academic Manuscripts and Artifacts

This directory contains the camera-ready papers, master's thesis, and presentation slide decks for the **NeuralOps** research project.

## Directory Structure & Build Protocols

Each academic subproject is authored in LaTeX and adheres to standard conference / institutional templates:

| Subproject | Source Path | Target Engine | Standard Build Command Sequence |
|---|---|---|---|
| **NeurIPS 2026 Paper** | `academic/paper/neurips2026-submission/` | `pdflatex` | `pdflatex neurips2026_conference.tex && bibtex neurips2026_conference && pdflatex neurips2026_conference.tex && pdflatex neurips2026_conference.tex` |
| **ICLR 2027 Paper** | `academic/paper/iclr2027-submission/` | `pdflatex` | `pdflatex iclr2027_conference.tex && bibtex iclr2027_conference && pdflatex iclr2027_conference.tex && pdflatex iclr2027_conference.tex` |
| **ArXiv Preprint** | `academic/paper/arxiv/` | `pdflatex` | `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` |
| **Master's Thesis** | `academic/thesis/` | `xelatex` | `xelatex -interaction=nonstopmode thesis.tex && bibtex thesis && xelatex thesis.tex && xelatex thesis.tex` |
| **Defense Slides** | `academic/slides/` | `xelatex` / Marp | `xelatex presentation.tex` |
| **Thesis Proposal** | `academic/proposal/` | `xelatex` | `xelatex thesis_proposal.tex` |

---

## Physical PDF Inventory

Empirical inventory of compiled academic artifacts present in the repository:

| Manuscript / Artifact | File Path | File Size | Page Count Status |
|---|---|---|---|
| **NeurIPS 2026 Conference Paper** | `academic/paper/neurips2026-submission/neurips2026_conference.pdf` | 64 MB | unverified (pdfinfo not available) |
| **Thesis Proposal** | `academic/proposal/فرم پروپوزال کارشناسی.pdf` | 344 KB | unverified (pdfinfo not available) |
| **Defense Presentation Deck** | `docs/presentation/810101504.pdf` | 7.0 MB | unverified (pdfinfo not available) |

---

## Build Integrity & Guardrails

1. **Clean Root Principle**: Never execute `pdflatex` or `xelatex` directly from the repository root. Always `cd` into the respective manuscript directory before compiling. This prevents dangling build artifacts (`texput.log`, `*.aux`, `*.out`) from polluting the git tree.
2. **Missing Section Guard**: Never include uncommitted scratch fragments (e.g. `\input{test_sec02.tex}`) in main build files without verifying their existence.
3. **Artifact Ignored Rule**: Compiled `.pdf` and `.log` outputs in `academic/` are managed via external release archives and gitignored to keep the git repository lightweight.
