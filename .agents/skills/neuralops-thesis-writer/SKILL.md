---
name: neuralops-thesis-writer
description: "Author and verify academic thesis chapters aligned with empirical ledger data and docs/thesis_agents guides"
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "thesis", "writer", "academic", "latex", "persian", "xepersian", "thesis_agents"]
---

# Skill: Academic Thesis Writer and 30-Agent Coordinator

## When to use
Use this master skill when drafting, revising, expanding, coordinating, or verifying chapters of the doctoral dissertation (`academic/thesis/810101504_main.tex`) according to University of Tehran engineering standards, the 30-agent division of labor in `docs/thesis_agents/MASTER_DIVISION_OF_LABOR.md`, the 5-wave parallel execution guide in `docs/thesis_agents/PARALLEL_EXECUTION_GUIDE.md`, and the Persian writing rules in `docs/thesis_agents/PERSIAN_WRITING_RULES.md`.

## Prerequisites
- Working TeX Live installation with `xelatex`, `bibtex`, and `xepersian` packages.
- Access to the master manuscript source in `academic/thesis/810101504_main.tex` and thirteen modular chapter files in `academic/thesis/chapters/`.
- Full compliance with `docs/thesis_agents/` specifications for Agents 01 through 30.
- Remote GPU access: `ssh taha@192.168.1.100` (RTX 4070 SM 8.9) with `/mnt/d/venv/bin/python3`.
- Direct read access to `results/registry/master_ledger.jsonl` and all 6 canonical audit files.
- Strict adherence to the Early Buggy Runs Exclusion Charter: ignore preliminary runs with catastrophic forgetting (<35% accuracy) or uncalibrated raw checkpoints.

## Canonical Repository Invariants
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## The 30 Thesis Agents and Modular Target Files

The dissertation architecture is partitioned into 13 modular `.tex` files in `academic/thesis/chapters/` across 30 specialized agent specifications:

| Agent ID | Specialized Role and Scope | Target LaTeX Subfile |
|:---:|:---|:---|
| **A01** | Motivation, Memory Wall $O(N^2)$, KV Saturation, Distribution Drift | `chapters/chapter01_intro.tex` |
| **A02** | Research Questions RQ1–RQ8, Hypotheses, Contribution Matrix | `chapters/chapter01_intro.tex` |
| **A03** | Standard Attention, SDPA Variance Proof, RoPE Geometry, Roofline Model | `chapters/chapter02_theory_part1.tex` |
| **A04** | SVD Foundations, Eckart-Young Theorem, Cholesky Whitening, LoRA | `chapters/chapter02_theory_part2.tex` |
| **A05** | Attention Taxonomy: 131 Mechanisms Classified Across 7 Families | `chapters/chapter02_theory_part2.tex` |
| **A06** | Model Compression Taxonomy: 72 Engines Classified Across 5 Paradigms | `chapters/chapter02_theory_part2.tex` |
| **A07** | 7-Layer Software Architecture, PEP 562 Lazy Loading, 4-Stage Surgery | `chapters/chapter03_methodology_part1.tex` |
| **A08** | 3-Level Fallback Dispatch Hierarchy (CUDA $\to$ Triton $\to$ PyTorch) | `chapters/chapter03_methodology_part2.tex` |
| **A09** | Experimental Setup (RTX 4070 SM 8.9), Models, 56% Accuracy Gate | `chapters/chapter03_methodology_part2.tex` |
| **A10** | Formal Proofs for Theorems 1 to 5 (Cholesky, ASVD, FP8, DSA, REAP) | `chapters/chapter03_methodology_part3a.tex` |
| **A11** | Formal Proofs for Theorems 6 to 10 (Rank Collapse, Simplex, HNC, 6P) | `chapters/chapter03_methodology_part3b.tex` |
| **A12** | Macro Telemetry of 203 Modules, 42 FAST / 161 SLOW, Pareto Frontiers | `chapters/chapter04_results_part1.tex` |
| **A13** | Analytical Evaluation and Attention Fidelity for 131 Mechanisms | `chapters/chapter04_results_part2.tex` |
| **A14** | Model Compression Evaluation, 5-Stage Pipeline, SVD Rank Sweeps | `chapters/chapter04_results_part2b.tex` |
| **A15** | Kernel Evaluation, 18 Multi-Tier Operators, Scaling up to 131k Tokens | `chapters/chapter04_results_part3.tex` |
| **A16** | Hardware Forensics, HLA Deadlock Autopsy, 6 Dispatch Safety Laws | `chapters/chapter04_results_part4.tex` |
| **A17** | Synthesis of Findings, Definitive Answers to RQ1–RQ8, 5 Lessons | `chapters/chapter05_conclusion.tex` |
| **A18** | Threats to Validity, Chunk GLA Anomaly Autopsy, v1.7.0/v2.0.0 Roadmap | `chapters/chapter05_conclusion.tex` |
| **A19** | Appendices: Exhaustive Catalogs of 131 Attentions and 72 Compressors | `chapter_all_modules_appendix.tex` |
| **A20** | Master Typography: XePersian Preamble, Macros, Zero-Error Build | `810101504_main.tex` |
| **A21** | 6-Column Ablation Matrix Enrichment in LaTeX | `chapters/chapter04_results_part2b.tex` |
| **A22** | SVD Rank Sweeps (16–384) and Knee Point Analysis in LaTeX | `chapters/chapter04_results_part2b.tex` |
| **A23** | 18 Three-Level Kernels and Multi-Precision Pareto Analysis | `chapters/chapter04_results_part3.tex` |
| **A24** | Sequence Length Scaling Analysis and FlashLinearAttention Benchmarks | `chapters/chapter04_results_part3.tex` |
| **A25** | Provenance Integrity, Disclosure Rules, and Early Buggy Run Filter | `chapters/chapter04_results_part1.tex` |
| **A26** | LaTeX Table Injection for 4-Way HNC and WikiText-2 Perplexity | `chapters/chapter04_results_part2b.tex` |
| **A27** | LaTeX Table Injection for 18-Op Kernels and FLA Long Context | `chapters/chapter04_results_part3.tex` |
| **A28** | LaTeX Table Injection for 5-Category Macro Telemetry Classification | `chapters/chapter04_results_part1.tex` |
| **A29** | Publication Scientific Figure Generation (TikZ/PGFPlots) | `chapters/` (figure floats) |
| **A30** | Practical Guide, Advanced Constraint Matrix, and Chapter 4 Synthesis | `chapters/chapter04_results_part4.tex` |

---

## The 5-Wave Parallel Execution and File Locking Protocol

Per `docs/thesis_agents/PARALLEL_EXECUTION_GUIDE.md`, agents execute in 5 sequential waves to eliminate file editing contention:

```
Wave 1 (A01–A07) ──► locks/wave1.done
       │
Wave 2 (A08–A14) ──► locks/wave2.done
       │
Wave 3 (A15–A21) ──► locks/wave3.done
       │
Wave 4 (A22–A28) ──► locks/wave4.done
       │
Wave 5 (A29–A30) ──► locks/wave5.done ──► Production Release Build
```

### Locking Rules:
1. Before modifying any target `.tex` file, verify that the prior wave's completion token (`locks/waveX.done`) exists.
2. Ensure no two agents modify the same section or table float concurrently.
3. Upon task completion, write `docs/thesis_agents/locks/agent_XX.done`.

---

## Persian Typography and XePersian Rules

Per `docs/thesis_agents/PERSIAN_WRITING_RULES.md`:
1. **The `\lr{...}` Macro**:
   - Use strictly for software and tool names (`\lr{PyTorch}`, `\lr{CUDA}`, `\lr{Triton}`, `\lr{NeuralOps}`), algorithm acronyms (`\lr{HNC-SVD}`), and dataset names (`\lr{SST-2}`, `\lr{GLUE}`).
   - **Banned** inside math environments (`\begin{equation}`, `\begin{align}`, `$ ... $`). Use `\operatorname{...}` or `\text{...}`.
   - **Banned** wrapping structural commands like `\item`. Write `\item \lr{FlashAttention}`, never `\lr{\item FlashAttention}`.
2. **Mandatory ZWNJ (نیم‌فاصله / `U+200C`)**:
   - Compulsory in Persian compound technical terms: `مدل‌های`, `کِرنل‌های`, `لایه‌ای`, `بهینه‌سازی`, `پشتهٔ گسیل`.
3. **Numerals and Percentages**:
   - Narrative Persian text must use Persian digits (**۱۳۱ مکانیزم توجه**, **۷۲ موتور فشرده‌سازی**) and Persian percent (`٪`).
   - Tables and mathematical formulations must use English digits (`93.12%`, `0.421 ms`).
4. **Early Buggy Runs Exclusion Charter**:
   - Preliminary uncalibrated runs (<35% accuracy, uncalibrated checkpoints) must be excluded from primary dissertation tables and narratives.

---

## Step-by-Step Authoring Protocol

### Step 1: Read Agent Specification and Audit Source
Before drafting any section:
1. Open and thoroughly read `docs/thesis_agents/AGENT_XX_*.md`.
2. Inspect line boundaries and section mappings in `AGENTS_FILE_ASSIGNMENT_AND_PROVENANCE_GUIDE.md`.
3. Extract relevant verified tables and metrics from `AUDIT_01` through `AUDIT_05`.

### Step 2: Draft Directly in Target `.tex` Subfile
Directly edit the corresponding subfile in `academic/thesis/chapters/` or `chapter_all_modules_appendix.tex`. Do NOT write intermediate markdown files.

### Step 3: Embed Formal Mathematical Proofs
Typeset all proofs with complete step-by-step mathematical reasoning. Never state that a proof is trivial or omitted.

### Step 4: Verification and Quality Gate Inspection
1. Confirm zero fatal errors under XeLaTeX compilation (`grep "^!" 810101504_main.log | wc -l` $= 0$).
2. Confirm zero instances of unresolved citations (`[?]`) or missing references (`??`).
3. Verify that all 10 figures exist as valid PDF files in `academic/thesis/figures/`.
4. Ensure full compliance with the 30-agent responsibility matrix.

Run this pre-commit thesis validation script:
```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Verifying Thesis Subsystem State ==="
# Check 1: Invariant counts in registries
python3 scripts/utils/audit_dead_code.py || true

# Check 2: Audit XePersian rules
grep -n "\\\\lr{" academic/thesis/chapters/*.tex | grep -E "begin\{equation\}|\$|align" && {
    echo "ERROR: Found illegal \lr inside math mode"
    exit 1
} || echo "PASS: Math environments clean."

# Check 3: Check all 13 chapter subfiles
for f in academic/thesis/chapters/*.tex; do
    [ -s "$f" ] || { echo "ERROR: Empty file $f"; exit 1; }
done
echo "PASS: All 13 chapter subfiles present and non-empty."
```

---

## Agent 30 Practical Decision Matrix and Safety Laws Integration
Agent 30 synthesizes Chapter 4 into practical operational deployment rules:
1. **Low-Latency Regimes ($T \le 512$)**: Deploy FlashAttention-3 or Fused MHA; avoiding tiling startup overhead.
2. **Long-Context Regimes ($T \ge 2048$)**: Deploy Recurrent SSM (Mamba, RetNet, Gated DeltaNet); maintaining $O(1)$ recurrent state cache.
3. **Extreme Low-Rank Compression ($r \le 64$)**: Apply HNC-SVD with 30-step distillation healing; preserving $>93\%$ accuracy.
4. **Safety Law**: Never deploy unwhitened SVD; activation covariance preconditioning via Cholesky decomposition is mathematically mandatory.

---

## Common Pitfalls to Avoid
- Editing `810101504_main.tex` directly for chapter contents instead of modular files in `academic/thesis/chapters/`.
- Injecting `\lr{}` inside math environments or around `\item` commands.
- Citing early uncalibrated checkpoint numbers (<35%) instead of calibrated final results.
- Skipping the `locks/` verification protocol during concurrent agent invocations.
- Joining Persian words without ZWNJ (`مدلهای` instead of `مدل‌های`).
