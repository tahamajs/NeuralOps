# FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention with Block Gating and Short-Sequence Fast Path

Paper package and modular publication draft for `fusion`.

## 📁 Directory Structure & Files

### 📑 Paper Sections (Modular Markdown)
Each section corresponds to a specific chapter/appendix of the paper draft:
- [`00_meta.md`](00_meta.md)
- [`01_intro.md`](01_intro.md)
- [`02_related.md`](02_related.md)
- [`03_background.md`](03_background.md)
- [`04_method.md`](04_method.md)
- [`05_architecture.md`](05_architecture.md)
- [`06_setup.md`](06_setup.md)
- [`07_results.md`](07_results.md)
- [`08_analysis.md`](08_analysis.md)
- [`09_discussion.md`](09_discussion.md)
- [`10_limitations.md`](10_limitations.md)
- [`11_conclusion.md`](11_conclusion.md)
- [`12_reproducibility.md`](12_reproducibility.md)
- [`13_references.md`](13_references.md)
- [`A_math.md`](A_math.md)
- [`B_implementation.md`](B_implementation.md)
- [`C_data.md`](C_data.md)
- [`D_negative_results.md`](D_negative_results.md)
- [`E_ethics.md`](E_ethics.md)
- [`checklist.md`](checklist.md) — Submission, honesty, and reproducibility checklist
- [`contents.md`](contents.md)
- [`paper_full.md`](paper_full.md) — **Full concatenated paper** compiled from all sections

### 🛠 Automation & Configuration Scripts
- [`apply_vars.py`](apply_vars.py) — Utility script (`build.py` compiles the full paper; `apply_vars.py` substitutes placeholders)
- [`build.py`](build.py) — Utility script (`build.py` compiles the full paper; `apply_vars.py` substitutes placeholders)
- [`variables.json`](variables.json) — Parameter and benchmark metrics dictionary for templating


### 📜 LaTeX Source Files (.tex)
This package includes a ready-to-compile modular LaTeX setup:
- [`main.tex`](main.tex) — Master LaTeX file importing all sections
- [`paper_full.tex`](paper_full.tex) — Self-contained standalone `.tex` document
- `00_meta.tex` through `13_references.tex` + Appendices `A_math.tex` through `E_ethics.tex`

Compile with `pdflatex` or `xelatex`:
```bash
pdflatex main.tex
```

## 🚀 Quick Usage

### 1. Build the Full Paper
To compile all markdown section files into a single unified `paper_full.md`:
```bash
python build.py
```
*(Optional: pass `--pdf` if `pandoc` and `xelatex` are installed to generate `paper_full.pdf`)*

### 2. Apply Template Variables
To inspect or substitute benchmark metrics from `variables.json`:
```bash
python apply_vars.py
```

### 3. Check Reproduction Status
See [`checklist.md`](checklist.md) and [`12_reproducibility.md`](12_reproducibility.md) before submitting or sharing the paper draft.
