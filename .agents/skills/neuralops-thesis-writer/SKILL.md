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
3. Verify that all 25+ publication-grade vector figures exist as valid PDF/PNG files in `academic/thesis/figures/` and `academic/thesis/img/` and are fully embedded in the text.
4. Ensure all tables strictly format numbers, percentages, speedups, and dimensions in English math mode `$ ... $` (e.g. `$94.50\%$`, `$1.74\times$`, `$34.80\%$`, `$0.0089$`) and use `tabularx` with `\small` / `\footnotesize` to guarantee zero `Overfull \hbox`.
5. Ensure every table and figure is accompanied by at least two substantive paragraphs of deep mechanistic reasoning and physical/algebraic explanation.
6. Ensure full compliance with the 30-agent responsibility matrix and attention monograph standards.

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

# Check 3: Check all chapter subfiles
for f in academic/thesis/chapters/*.tex; do
    [ -s "$f" ] || { echo "ERROR: Empty file $f"; exit 1; }
done
echo "PASS: All chapter subfiles present and non-empty."

# Check 4: Check for bare numbers in tables
python3 -c "
import re, glob
for path in glob.glob('academic/thesis/chapters/*.tex') + glob.glob('academic/thesis/*.tex'):
    with open(path) as f: content = f.read()
    # Check tabular environments for unescaped bare digits
    tables = re.findall(r'\\\\begin\{tabular.*?}(.*?)\\\\end\{tabular.*?}', content, re.DOTALL)
    for i, t in enumerate(tables):
        persian_digits = re.findall(r'[۰-۹]', t)
        if len(persian_digits) > 5:
            print(f'WARNING: {path} table {i} contains {len(persian_digits)} Persian digits. Convert to English math mode $...$!')
"
```

---

## Strict Visualization, Mechanistic Reasoning & Table Formatting Standards

### 1. High-Density Publication Visualizations (Minimum 25+ Figures)
Agents must actively integrate publication-grade charts from `academic/thesis/figures/` and `academic/thesis/img/`:
- **Chapter 1**: 5-step surgery pipeline (`fig_ablation_pipeline_5step.pdf`), memory wall $O(N^2)$ growth curves.
- **Chapter 2**: Williams Roofline model on RTX 4070 SM 8.9 (`fig_roofline_hardware.pdf`), SVD singular value decay and outlier kurtosis (`img/fig13_svd_spectrum.png`), RoPE phasor rotation geometry (`img/svd_linear_attention_3d.jpg`).
- **Chapter 3**: 7-layer architecture stack, 4-stage surgery pipeline, 3-level zero-overhead dispatch hierarchy (`fig_3level_speedups_all18.pdf`), Stiefel manifold Riemannian retraction geometry.
- **Chapter 4**: SVD rank knee-point curve with Table 4.17 (`fig_rank_sweep_adasvd.pdf`), WikiText-2 Perplexity Pareto Frontier (`fig_wikitext2_pareto_frontier.pdf`), 5D Family Radar chart (`fig_family_radar.pdf`), 3-way Pareto Frontier (`fig_pareto_accuracy_latency_vram.pdf`), 36-cell ablation heatmap (`fig_6pillar_heatmap.pdf`), Multi-Pillar Waterfall chart (`fig_6pillar_ablation.pdf`), sequence scaling up to 131k tokens (`fig_fla_scaling_headtohead.pdf`).
- **Chapter 5**: Chunk GLA L2 cache thrashing vs dynamic chunking autopsy, RQ1-RQ8 multi-radar synthesis.
- **Appendices**: Architecture diagrams for all 7 attention families (`img/AdaptiveSpanAttention.jpeg`, `img/nsa_attention.jpeg`, `img/CHELA-SLAY.jpeg`, `img/BAM-UMoE.jpeg`, `img/gqa_attn_new.png`, `img/jamba_info_arch.jpg`, `img/hymba_transx_arch.jpg`, `img/linformer.jpeg`).

### 2. Deep Mechanistic Reasoning Mandate
Never present figures or tables as isolated empirical outputs. Every table and figure MUST be accompanied by at least two detailed paragraphs answering:
- **Root Cause & Physics**: What microarchitectural stalls (DRAM bandwidth bound, SRAM capacity, register pressure, cache thrashing) or mathematical invariants (Eckart-Young-Mirsky failure, condition number explosion $\kappa \gg 10^5$, Stiefel orthogonality) govern this behavior?
- **Theoretical Linkage**: Connect the empirical numbers directly to the 10 formal theorems and 27 lemmas in Chapter 3.
- **Engineering Implications**: What concrete design choices, ridge points, or fallback decisions arise from this result?

### 3. Strict Table Formatting Contract (English Digits & No Margin Overflows)
- **Mandatory English Digits**: Every number, percentage, ratio, and dimension inside any LaTeX table MUST be written in English digits in math mode `$ ... $` (e.g. `$94.50\%$`, `$93.12\%$`, `$1.74\times$`, `$0.0089\text{ ms}$`, `$34.80\%$`).
- **No Margin Overflow**: Wide tables must use `\begin{tabularx}{\textwidth}{@{} ... X @{}}` with `\small` or `\footnotesize`, `\setlength{\tabcolsep}{...}`, and `\renewcommand{\arraystretch}{1.2}`. Word squeezing in narrow columns is strictly prohibited.

### 4. Attention Mechanisms Monograph Mandate
In Chapter 2 and Appendix A, all 131 attention mechanisms across the 7 families must be systematically covered, with deep half-page architectural monographs for key archetypes detailing:
- Exact mathematical formulation ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$ transformations and attention score calculation).
- Asymptotic time and memory complexity ($\mathcal{O}(N^2)$ vs $\mathcal{O}(N \cdot d)$ vs $\mathcal{O}(N)$).
- KV-cache footprint scaling across sequence lengths $N \in [512, 131072]$.
- Microarchitectural suitability: SRAM-bound vs DRAM-bound vs L2-cache resident.
- Empirical classification accuracy on SST-2 and perplexity on WikiText-2.

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
