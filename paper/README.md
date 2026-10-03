# 📄 NeuralOps Publications & Academic Artifacts

Welcome to the **NeuralOps Academic Repository**! 🎓 This directory serves as the centralized hub for all conference manuscripts, camera-ready LaTeX sources, bibliography references, and individual module papers related to the **NeuralOps** framework.

---

## 📂 Directory Structure

Our publications and reports are meticulously organized into the following areas:

- 🌟 **`iclr2027-submission/`**: Primary conference paper targeting ICLR 2027 (`iclr2027_conference.tex` and compiled `iclr2027_conference.pdf`).
- 🖼️ **`iclr2027-abstract/`**: Extended abstract and visual poster materials.
- 📚 **`PerModulePaper/`**: 217 individual standalone conference papers (each with its own `main.tex`, figures, and compiled `main.pdf`).
- 📜 **`references_arxiv/`**: Verified BibTeX entries and arXiv reference PDFs.
- 🎨 **`template/`**: ICLR conference formatting style sheets and macros.
- 📊 **`reports/`**: Architectural and empirical diagnostic reports for evaluated modules.
- 🔍 **`audit/`**: Citation integrity audits, numerical checks, and reproducibility logs.
- 🧪 **`experiments/`**: Parameter sweeps, compile logs, and fairness calibration traces.

---

## 🛠️ Main Compilation (ICLR 2027)

To compile the primary conference paper, you can use the standard LaTeX toolchain:

```bash
cd iclr2027-submission
pdflatex -interaction=nonstopmode iclr2027_conference.tex
bibtex iclr2027_conference
pdflatex -interaction=nonstopmode iclr2027_conference.tex
pdflatex -interaction=nonstopmode iclr2027_conference.tex
```

## 🏗️ Building Module Papers

You can use the automated `build_paper.py` script to seamlessly concatenate modular markdown sections into unified paper drafts or PDFs.

**Build a specific module paper:**
```bash
python build_paper.py --module vspec
```

**Build all module papers:**
```bash
python build_paper.py --all
```

---
*Generated and maintained as part of the NeuralOps Reproducibility Pipeline.* 🚀
