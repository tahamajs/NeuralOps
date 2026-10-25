---
name: report-latex-compiler
description: "Compile the complete thesis/report with xelatex and ensure 0 errors, 0 warnings"
version: "1.0.0"
tags: ["neuralops", "report", "latex", "compilation", "xelatex", "bibtex"]
---

# Skill: Production LaTeX Compilation and Diagnostic Resolution

## When to use
Use this skill when compiling the complete doctoral thesis manuscript, technical monograph, or conference paper into its final distribution PDF format. It governs the multi-pass compilation protocol using `xelatex` and `bibtex`, enforces zero-error quality gates, diagnoses typesetting anomalies, and verifies page count and bibliography resolution.

## Prerequisites
- TeX Live distribution with `xelatex`, `bibtex`, and standard utility `pdfinfo`.
- Local manuscript workspace root: `thesis_final/` containing `810101504_main.tex` and `references.bib`.
- Installed system fonts: Latin Modern, Computer Modern, or TeX Gyre Termes.
- Executable bash environment for running compilation automation scripts.

## Canonical Audit Files Reference
The compilation pipeline integrates all sections and bibliography citations pointing back to the six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced in bibliography entry `\cite{audit01}` for distillation recovery results and Ridge calibration tables.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced in bibliography entry `\cite{audit02}` for SVD rank sweep matrices and Cholesky whitening proofs.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced in bibliography entry `\cite{audit03}` for 203 unified module benchmarks and roofline evaluations.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced in bibliography entry `\cite{audit04}` for sequence scaling logarithmic plots and KV cache memory tables.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced in bibliography entry `\cite{audit05}` for testbed execution disclosures and Git commit SHA provenance.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Sourced in bibliography entry `\cite{agents_guide}` for chapter assignment governance and author ownership records.

---

## The Canonical Multi-Pass Compilation Workflow

LaTeX requires multiple passes to resolve forward references, generate table of contents entries, and link external BibTeX citations:

```bash
cd thesis_final/

# 1. Purge stale build artifacts to prevent cache corruption
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lot *.lof *.nav *.snm

# 2. First Pass: Parse document structure and generate initial .aux tables
xelatex -interaction=nonstopmode 810101504_main.tex

# 3. Bibliography Pass: Compile external references and link audit citations
bibtex 810101504_main

# 4. Second Pass: Resolve citation keys and internal cross-references (\ref, \cite)
xelatex -interaction=nonstopmode 810101504_main.tex

# 5. Third Pass: Finalize Table of Contents, List of Figures, and page numbers
xelatex -interaction=nonstopmode 810101504_main.tex

# 6. Verify compilation metrics
pdfinfo 810101504_main.pdf | grep Pages
grep -E "^\!|Warning" 810101504_main.log | wc -l
```

---

## Strict Success Criteria

A build is certified as **PASS** only if ALL of the following conditions are met:
1. `810101504_main.pdf` exists and contains $> 100$ pages.
2. `grep "^!" 810101504_main.log` returns exactly `0` fatal errors.
3. Overfull and underfull warnings are strictly minimized (`grep "Overfull\|Underfull" 810101504_main.log | wc -l` $< 10$).
4. Zero unresolved citations: scanning the compiled PDF text reveals zero instances of `[?]` or `[??]`.
5. Zero dangling internal cross-references: scanning reveals zero instances of `??` in text floats.

---

## Diagnostic Triage: Common LaTeX Errors and Solutions

| Symptom / Compiler Error Message | Underlying Root Cause | Verified Canonical Resolution |
|:---|:---|:---|
| `Undefined control sequence \softmax` | Missing custom math operator in LaTeX preamble | Add `\DeclareMathOperator{\softmax}{softmax}` in `preamble.tex` |
| `Command \lr invalid in math mode` | Unwrapped text string inside an equation | Wrap variable in `\text{lr}` or `\mathit{lr}` |
| `Missing $ inserted` | Unescaped special character (`_`, `%`, `&`, `#`) | Escape symbol with backslash (`\_`, `\%`, `\&`, `\#`) |
| `Citation 'audit01' undefined` | Missing entry in `references.bib` or skipped `bibtex` | Verify bib entry and re-run full 3-pass sequence |
| `Package longtable Error: Table head ...` | Misplaced `\endhead` or missing table delimiters | Ensure `\endhead` follows the initial header row |
| `Overfull \hbox (24.2pt too wide)` | Unbreakable monospace identifier in narrow column | Wrap path in `\seqsplit{...}` or allow line breaks |
| `Dimension too large` | PGF/TikZ coordinate out of allowable bounds | Rescale figure coordinates or use pre-rendered vector PDF |
| `Package fontspec Error: Font cannot be found` | System font name mismatch in XeLaTeX setup | Switch fontspec to Latin Modern or installed TrueType/OpenType font |
| `Too many math alphabets used in version normal` | Too many specialized fonts loaded simultaneously | Consolidate math packages and remove redundant font calls |
| `Float(s) lost` | Excess unplaced figures exceeding LaTeX float buffer | Insert `\clearpage` or load `morefloats` package in preamble |

---

## Preamble Configuration Standards for XeLaTeX

Ensure the master document preamble (`thesis_final/preamble.tex`) includes required packages to avoid compilation regressions:

```latex
\usepackage{fontspec}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{float}
\usepackage{seqsplit}
\usepackage{microtype}
\usepackage{hyperref}

% Declare required math operators
\DeclareMathOperator{\softmax}{softmax}
\DeclareMathOperator{\sigmoid}{sigmoid}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\tr}{tr}

% Theorem environments
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}
```

---

## Automated Compilation Harness (`scripts/utils/verify_thesis_compile.sh`)

Deploy the following automated shell verification script to enforce build standards across continuous integration and release checks:

```bash
#!/bin/bash
set -eo pipefail

cd thesis_final/
echo "=== Starting Clean Compilation Pass for NeuralOps Thesis ==="
rm -f *.aux *.log *.out *.toc *.bbl *.blg

xelatex -interaction=nonstopmode 810101504_main.tex > /dev/null 2>&1 || true
bibtex 810101504_main > /dev/null 2>&1 || true
xelatex -interaction=nonstopmode 810101504_main.tex > /dev/null 2>&1 || true
xelatex -interaction=nonstopmode 810101504_main.tex > /dev/null 2>&1 || true

ERRORS=$(grep "^!" 810101504_main.log | wc -l || true)
PAGES=$(pdfinfo 810101504_main.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "0")
DANGLING_CITES=$(pdftotext 810101504_main.pdf - 2>/dev/null | grep -c "\[\?\?\]" || true)

echo "Compilation Summary:"
echo "  Fatal Errors:     $ERRORS"
echo "  Total Pages:      $PAGES"
echo "  Dangling Cites:   $DANGLING_CITES"

if [ "$ERRORS" -eq 0 ] && [ "$PAGES" -gt 100 ] && [ "$DANGLING_CITES" -eq 0 ]; then
    echo "Result: PASS (Thesis compilation certified)"
    exit 0
else
    echo "Result: FAIL (Quality gates not met)"
    exit 1
fi
```

---

## Common Pitfalls
- Running `xelatex` only once, leaving citations, figures, and table of contents unresolved.
- Ignoring fatal errors hidden when running in `-interaction=nonstopmode` without inspecting `.log`.
- Overlooking unescaped underscores (`_`) in module names (e.g., writing `flash_attention_v2` instead of `\texttt{flash\_attention\_v2}`).
- Failing to verify font embedding, which causes text corruption in PDF print production.
- Disregarding author metadata declared in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.
