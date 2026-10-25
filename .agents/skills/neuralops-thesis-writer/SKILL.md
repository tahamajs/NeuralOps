---
name: neuralops-thesis-writer
description: "Author and verify academic thesis chapters aligned with empirical ledger data and docs/thesis_agents guides"
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "thesis", "writer", "academic", "latex", "persian", "xepersian"]
---

# Skill: Academic Thesis Writer and Thesis Agent Coordinator

## When to use
Use this skill when drafting, revising, expanding, or compiling chapters of the academic doctoral dissertation (`academic/thesis/810101504_main.tex` or `thesis_final/810101504_main.tex`) according to University of Tehran standards.

## Prerequisites
- Working TeX Live installation with `xelatex`, `bibtex`, and `xepersian` packages.
- Access to the master manuscript source and `references.bib`.
- Remote GPU access: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9) for telemetry verification.
- Read access to `results/registry/master_ledger.jsonl`.

## Canonical Audit Files and Thesis Agent Specifications Reference
Every section must strictly follow its assigned specification in `docs/thesis_agents/`:

1. **`MASTER_DIVISION_OF_LABOR.md`**: Master division of labor across all 20 core thesis agents and chapter assignments.
2. **`AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`**: Line numbers in `810101504_main.tex`, provenance keys, and audit table mappings.
3. **`PERSIAN_WRITING_RULES.md`**: University of Tehran official Persian academic writing rules, XePersian constraints, and typography conventions.
4. **Agent-Specific Directives (`AGENT_01_*.md` through `AGENT_30_*.md`)**:
   - `AGENT_01_CHAPTER_1_FOUNDATIONS.md`: Motivation, memory explosion, and memory wall analysis.
   - `AGENT_02_CHAPTER_1_RQS_CONTRIBUTIONS.md`: Research questions RQ1–RQ8, hypotheses, and contributions.
   - `AGENT_03_CHAPTER_2_SDPA_ROOFLINE.md`: Classic attention mechanisms, Flash tiling, and hardware roofline.
   - `AGENT_04_CHAPTER_2_SVD_CHOLESKY.md`: SVD low-rank theory and Cholesky covariance whitening.
   - `AGENT_05_CHAPTER_2_ATTENTION_TAXONOMY.md`: Categorization of all 131 attention mechanisms across 7 families.
   - `AGENT_06_CHAPTER_2_COMPRESSION_PARADIGMS.md`: Categorization of all 72 compression engines across 5 paradigms.
   - `AGENT_07_CHAPTER_3_ARCHITECTURE_SURGERY.md`: 7-layer architecture and 4-stage surgical transplant protocol.
   - `AGENT_08_CHAPTER_3_DISPATCH_CORRECTNESS.md`: 3-level fallback hierarchy and numerical correctness gates.
   - `AGENT_09_CHAPTER_3_EXPERIMENTAL_SETUP.md`: Testbed setup (RTX 4070 SM 8.9), data pipelines, and baselines.
   - `AGENT_10_CHAPTER_3_THEOREMS_1_TO_5.md`: Formal proofs for Theorems 1 through 5.
   - `AGENT_11_CHAPTER_3_THEOREMS_6_TO_10.md`: Formal proofs for Theorems 6 through 10.
   - `AGENT_12_CHAPTER_4_MACRO_TELEMETRY.md`: Macro 203-module telemetry evaluation and Pareto frontiers.
   - `AGENT_13_CHAPTER_4_ATTENTION_EVALUATION.md`: Performance breakdown across the 131 attention mechanisms.
   - `AGENT_14_CHAPTER_4_COMPRESSION_DISTILLATION.md`: 5-stage ablation study and 6-pillar distillation regimes.
   - `AGENT_15_CHAPTER_4_KERNELS_SCALING.md`: Sequence length scaling ($N \in [128, 8192]$) and memory caching.
   - `AGENT_16_HARDWARE_FORENSICS_SAFETY.md`: The 6 deep hardware forensics case studies and safety mitigations.
   - `AGENT_17_CHAPTER_5_SYNTHESIS_ANSWERS.md`: Empirical synthesis and answers to research questions RQ1–RQ8.
   - `AGENT_18_CHAPTER_5_VALIDITY_ROADMAP.md`: The 7 threats to validity and future release roadmap.
   - `AGENT_19_APPENDICES_CATALOG_203.md`: Full catalogs of 131 attentions, 72 compressors, and 108 CUDA kernels.
   - `AGENT_20_TYPOGRAPHY_COMPILATION_BIB.md`: Compilation validation, bibliography resolution, and typography.
   - `AGENT_21`–`AGENT_30`: Audit enrichment and practical LaTeX chapter guides.
5. **Canonical Audit Files**: `AUDIT_01` through `AUDIT_05` and `results/registry/master_ledger.jsonl`.

---

## Step-by-Step Execution Protocol

### Step 1: Ingest Assigned Agent Guide
Before writing any paragraph or table:
1. Identify the chapter and section being authored.
2. Read the corresponding `docs/thesis_agents/AGENT_XX_*.md` file completely.
3. Check line boundaries and section labels in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.

### Step 2: Enforce XePersian and Persian Academic Rules
Follow `PERSIAN_WRITING_RULES.md`:
- **`\lr{...}` Macro**:
  - Use strictly for software/tool names (`\lr{PyTorch}`, `\lr{CUDA}`, `\lr{Triton}`, `\lr{NeuralOps}`), algorithm acronyms (`\lr{HNC-SVD}`), and dataset names (`\lr{SST-2}`, `\lr{GLUE}`).
  - **Banned** inside math mode (`\begin{equation}` / `$ ... $`). Use `\operatorname{...}` or `\text{...}`.
  - **Banned** enclosing `\item`. Never write `\lr{\item ...}`.
- **نیم‌فاصله (ZWNJ / `U+200C`)**:
  - Mandatory in compound Persian terms: `مدل‌های`, `کِرنل‌های`, `لایه‌ای`, `بهینه‌سازی`.
- **Persian Digits & Percentage**:
  - In narrative Persian text, write numbers in Persian digits (**۱۳۱ مکانیزم توجه**, **۷۲ موتور فشرده‌سازی**).
  - In Persian sentences, use `٪`, never English `%`.

### Step 3: Integrate Empirical Telemetry & Proofs
- Quote exact numerical metrics from the master ledger (`results/registry/master_ledger.jsonl`) and `AUDIT_01`–`AUDIT_05`.
- Typeset complete proofs for Theorems 1–10 without steps omitted.

### Step 4: Multi-Pass Compilation Verification
Compile cleanly with XeLaTeX:
```bash
cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex && bibtex 810101504_main && xelatex -interaction=nonstopmode 810101504_main.tex
```

---

## Verification
- Confirm 0 fatal errors (`grep "^!" 810101504_main.log | wc -l` $= 0$).
- Confirm 0 instances of `TODO`, `TBD`, `???`, or placeholder numbers.
- Confirm all citations and section labels resolve cleanly.

---

## Common Pitfalls
- Writing chapters without consulting the designated `docs/thesis_agents/AGENT_XX_*.md` guide.
- Misusing `\lr{...}` inside LaTeX math environments, causing fatal compilation errors.
- Joining Persian words without ZWNJ (`مدلهای` instead of `مدل‌های`).
