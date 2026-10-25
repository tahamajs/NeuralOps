---
name: report-latex-compiler
description: "Compile the complete thesis/report with xelatex and ensure 0 errors, 0 warnings per docs/thesis_agents"
version: "1.0.0"
tags: ["neuralops", "report", "latex", "compilation", "xelatex", "bibtex", "thesis_agents"]
---

# Skill: Production LaTeX Compilation and Diagnostic Resolution

## When to use
Use this skill when compiling the complete doctoral thesis manuscript, technical monograph, or academic defense document (`academic/thesis/810101504_main.tex`). It governs the multi-pass compilation protocol using `xelatex` and `bibtex`, enforces zero-error quality gates, audits XePersian typesetting anomalies, and verifies bibliography resolution according to Agent 20.

## Prerequisites
- Direct compliance with `docs/thesis_agents/AGENT_20_TYPOGRAPHY_COMPILATION_BIB.md`.
- Full compliance with `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md` and `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.
- Target files: `academic/thesis/810101504_main.tex` and all thirteen modular files in `academic/thesis/chapters/`.
- TeX Live distribution with `xelatex`, `bibtex`, and `xepersian` package.
- System Persian and Latin fonts configured in `academic/thesis/tehran-thesis.cls`.

## Canonical Audit Files Reference
The compilation pipeline integrates all sections and bibliography citations pointing back to the six canonical audit documents:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced in bibliography entry `\cite{audit01}` for distillation recovery results and Ridge calibration tables.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced in bibliography entry `\cite{audit02}` for SVD rank sweep matrices and Cholesky whitening proofs.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced in bibliography entry `\cite{audit03}` for 203 unified module benchmarks and roofline evaluations.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced in bibliography entry `\cite{audit04}` for sequence scaling logarithmic plots and KV cache memory tables.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced in bibliography entry `\cite{audit05}` for testbed execution disclosures and Git commit SHA provenance.
6. `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`: Sourced in bibliography entry `\cite{agents_guide}` for chapter assignment governance and author ownership records.

## Canonical Repository Invariants
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Master Document Architecture and Modular Inclusions (Agent 20)
Target: `academic/thesis/810101504_main.tex`.
Verify that `810101504_main.tex` includes all 13 modular chapter components in correct execution order:

```latex
% فراخوانی فصول ماژولار سیزده‌گانه
\input{chapters/chapter01_intro.tex}
\input{chapters/chapter02_theory_part1.tex}
\input{chapters/chapter02_theory_part2.tex}
\input{chapters/chapter03_methodology_part1.tex}
\input{chapters/chapter03_methodology_part2.tex}
\input{chapters/chapter03_methodology_part3a.tex}
\input{chapters/chapter03_methodology_part3b.tex}
\input{chapters/chapter04_results_part1.tex}
\input{chapters/chapter04_results_part2.tex}
\input{chapters/chapter04_results_part2b.tex}
\input{chapters/chapter04_results_part3.tex}
\input{chapters/chapter04_results_part4.tex}
\input{chapters/chapter05_conclusion.tex}
```

---

## Master Preamble & XePersian Typography Configuration

```latex
% ============================================================================
% تعاریف اپراتورهای استاندارد ریاضی جهت جلوگیری از خطای \undefined
% ============================================================================
\DeclareMathOperator{\softmax}{softmax}
\DeclareMathOperator{\Softmax}{Softmax}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\TopK}{TopK}
\DeclareMathOperator{\clamp}{clamp}
\DeclareMathOperator{\polar}{polar}

% تصحیح علائم پرانتز شماره فرمول‌ها در زی‌پرشین
\makeatletter
\def\tagform@#1{\maketag@@@{\lr{(}\ignorespaces#1\unskip\@@italiccorr\lr{)}}}
\makeatother

% رفع تداخل موقعیت‌یابی متن در تیکز در محیط زی‌پرشین
\def\UseMathForPositioningText{}
```

---

## Canonical Multi-Pass Compilation Sequence

```bash
cd academic/thesis/

# 1. پاکسازی فایل‌های موقت قدیمی جهت جلوگیری از تداخل حافظه کش
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lot *.lof *.nav *.snm chapters/*.aux

# 2. گام اول: تحلیل ساختار سند و ایجاد فایل‌های .aux
xelatex -interaction=nonstopmode 810101504_main.tex

# 3. گام مراجع: اجرای موتور BibTeX برای اتصال ارجاعات مقالات و پرونده‌های ممیزی
bibtex 810101504_main

# 4. گام دوم: اتصال برچسب‌های متقابل (\ref و \cite)
xelatex -interaction=nonstopmode 810101504_main.tex

# 5. گام سوم: نهایی‌سازی فهرست مطالب، فهرست اشکال، جداول و شماره صفحات
xelatex -interaction=nonstopmode 810101504_main.tex

# 6. بررسی تله‌متری و کیفیت خروجی کامپایل
pdfinfo 810101504_main.pdf | grep Pages
grep -E "^\!|Warning" 810101504_main.log | wc -l
```

---

## Strict Success Criteria (Agent 20 Quality Gates)

A build is certified as **PASS** only if ALL conditions are met:
1. `810101504_main.pdf` is generated successfully with Exit Code 0.
2. `grep "^!" 810101504_main.log` returns exactly `0` fatal errors.
3. Zero unresolved citation markers (`[?]` or `[??]`) in compiled output.
4. Zero missing label references (`??` in cross-references).
5. Zero unescaped special characters (`_`, `%`, `&`, `#`) breaking font encoding.
6. Zero instances of `\lr{}` inside math environments or wrapping `\item`.

---

## Diagnostic Triage for XePersian Compilation Anomalies

| Compiler Error / Diagnostic Message | Underlying Cause | Canonical Resolution |
|:---|:---|:---|
| `Undefined control sequence \softmax` | Missing operator declaration | Add `\DeclareMathOperator{\softmax}{softmax}` in preamble |
| `Command \lr invalid in math mode` | `\lr{}` injected into equation | Replace with `\text{...}` or `\operatorname{...}` |
| `! LaTeX Error: Something's wrong--perhaps a missing \item` | `\item` wrapped inside `\lr{...}` | Move `\item` outside: `\item \lr{FlashAttention}` |
| `Package fontspec Error: Font cannot be found` | System font mismatch | Ensure IRNazanin, Vazirmatn, or XB Niloofur is registered in fontspec |
| `Dimension too large in pgfplots` | TikZ coordinate out of bounds | Clip plot domain or use pre-rendered vector PDF |
| `Overfull \hbox (24.2pt too wide)` | Long unbroken monospace token | Wrap token in `\seqsplit{...}` or allow soft hyphens |
| `Citation 'audit01' undefined` | Stale aux file or missing bibtex | Run complete 3-pass compilation sequence |

---

## Automated Verification Script

Run the automated pre-flight audit script to verify compilation readiness:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Verifying XePersian Syntax and Invariants ==="

# 1. Audit for illegal \lr inside math environments
ILLEGAL_LR=$(grep -n "\\\\lr{" academic/thesis/chapters/*.tex | grep -E "begin\{equation\}|\$|align" || true)
if [ -n "$ILLEGAL_LR" ]; then
    echo "ERROR: Found illegal \lr inside math environment:"
    echo "$ILLEGAL_LR"
    exit 1
fi

# 2. Audit for illegal \lr around \item
ILLEGAL_ITEM=$(grep -n "\\\\lr{.*\\\\item" academic/thesis/chapters/*.tex || true)
if [ -n "$ILLEGAL_ITEM" ]; then
    echo "ERROR: Found \item wrapped inside \lr{}:"
    echo "$ILLEGAL_ITEM"
    exit 1
fi

# 3. Check for unescaped underscores
UNESCAPED_UNDERSCORE=$(grep -n "[^\\]_" academic/thesis/chapters/*.tex | grep -v "texttt" | grep -v "label" | grep -v "ref" | grep -v "\$" || true)
if [ -n "$UNESCAPED_UNDERSCORE" ]; then
    echo "WARNING: Check for potential unescaped underscores:"
    echo "$UNESCAPED_UNDERSCORE" | head -n 5
fi

echo "All typography pre-flight checks passed."
```

---

## Common Pitfalls to Avoid
- Running single-pass `xelatex` and mistaking citation warnings for fatal syntax errors.
- Placing `\lr{}` around LaTeX structural keywords (`\item`, `\begin`, `\end`).
- Forgetting to purge `.aux` files after renaming `\label{...}` targets.
- Inserting raw English text without `\lr{}` in narrative Persian paragraphs.
- Allowing unescaped underscores (`_`) in file paths or registry names outside `\texttt{}`.
