---
name: latex-compilation-diagnostics
description: "Diagnose, isolate, and resolve complex XeLaTeX compilation errors, font conflicts, and float overflows"
version: "1.0.0"
tags: ["neuralops", "latex", "compiler", "debugging", "diagnostics"]
---

# Skill: LaTeX Compilation Diagnostics and Automated Triage

## When to use
Use this skill when encountering build failures, missing font errors, table formatting exceptions, or unresolved citation warnings during manuscript compilation in `thesis_final/` or `paper/`. It provides systematic root-cause isolation and automated fixes.

## Prerequisites
- Manuscript workspace: `thesis_final/` containing `810101504_main.tex` and `references.bib`.
- Remote GPU host: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9).
- Master results ledger: `results/registry/master_ledger.jsonl`.
- System tools: `xelatex`, `bibtex`, `pdfinfo`, `grep`, `sed`.

## Canonical Audit Files Reference
Diagnostics must preserve all citation links to the six canonical audits:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Linked via `\cite{audit01}` for distillation recovery results.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Linked via `\cite{audit02}` for low-rank SVD benchmarks.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Linked via `\cite{audit03}` for 203 unified module latencies and roofline data.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Linked via `\cite{audit04}` for sequence scaling arrays.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Linked via `\cite{audit05}` for testbed execution parameters.

## Canonical Repository Invariants
Ensure manuscript preserves canonical counts:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Diagnostic Workflow and Automated Triage

### 1. Log File Inspection Engine
Execute a fast log scraper to extract compiler errors and their file line coordinates:
```bash
grep -n -C 3 "^!" thesis_final/810101504_main.log || echo "No fatal errors located."
```

### 2. Systematic Error Classification & Resolution
- **Unescaped TeX Characters (`_`, `%`, `&`)**:
  *Symptom*: `Missing $ inserted` or unexpected comment truncation.
  *Fix*: Replace with `\_`, `\%`, `\&` in module identifiers.
- **Dangling Cross-References**:
  *Symptom*: `LaTeX Warning: Reference 'fig:xyz' on page X undefined`.
  *Fix*: Run the complete 3-pass compilation sequence (`xelatex` $\rightarrow$ `bibtex` $\rightarrow$ `xelatex` $\rightarrow$ `xelatex`).
- **Longtable Alignment Errors**:
  *Symptom*: `Package longtable Error: Table head not finished with \endhead`.
  *Fix*: Insert `\endhead` immediately after the first header block.
- **Fontspec Glyphs Missing**:
  *Symptom*: `Font \TU/lmr/m/n/10=nullfont not loadable`.
  *Fix*: Fallback fontspec configuration to standard Latin Modern (`\setmainfont{Latin Modern Roman}`).

---

## Verification
Confirm that the document compiles without errors:
1. **Invariant Counts**:
   ```bash
   python3 scripts/utils/verify_readme_counts.py
   # Must confirm 131 / 72 / 26
   ```
2. **Audit Citations**:
   ```bash
   grep -c "AUDIT_0" thesis_final/810101504_main.tex
   # Must be >= 5
   ```
3. **Clean Compilation Check**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must exit with code 0 and 0 fatal errors
   ```

---

## Common Pitfalls
- Deleting entire equations or tables to suppress compiler errors rather than diagnosing the underlying syntax bug.
- Ignoring Overfull `\hbox` warnings where code paths spill into the page margins.
- Omitting the `bibtex` pass when debugging citation undefined warnings.
