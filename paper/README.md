# NeuralOps & SVD Linear Attention: Publication Repository

This repository contains all conference manuscripts, camera-ready LaTeX sources, bibliography references, and individual module papers.

## Directory Structure

- **`iclr2027-submission/`**: Primary conference paper targeting ICLR 2027 (`iclr2027_conference.tex` and compiled `iclr2027_conference.pdf`).
- **`iclr2027-abstract/`**: Extended abstract and visual poster materials.
- **`PerModulePaper/`**: 217 individual standalone conference papers (each with its own `main.tex`, figures, and compiled `main.pdf`).
- **`references_arxiv/`**: Verified BibTeX entries and arXiv reference PDFs.
- **`template/`**: ICLR conference formatting style sheets and macros.
- **`reports/`**: Architectural and empirical diagnostic reports for evaluated modules.
- **`audit/`**: Citation integrity audits, numerical checks, and reproducibility logs.
- **`experiments/`**: Parameter sweeps, compile logs, and fairness calibration traces.

## Main Compilation
To compile the primary conference paper:
```bash
cd iclr2027-submission
pdflatex -interaction=nonstopmode iclr2027_conference.tex
bibtex iclr2027_conference
pdflatex -interaction=nonstopmode iclr2027_conference.tex
pdflatex -interaction=nonstopmode iclr2027_conference.tex
```
