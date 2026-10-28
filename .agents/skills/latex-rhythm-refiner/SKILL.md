---
name: latex-rhythm-refiner
description: "Refine prose rhythm, paragraph cohesion, and sentence cadence while preserving LaTeX syntax and citations"
version: "1.0.0"
tags: ["neuralops", "latex", "editing", "style", "rhythm"]
---

# Skill: LaTeX Prose Rhythm and Cohesion Refiner

## When to use
Use this skill when polishing dense academic sections of the doctoral thesis or conference papers. It optimizes sentence cadence, introduces dynamic syntactic variation, and enhances paragraph-to-paragraph transitions while rigorously protecting embedded LaTeX environments, citation tags, and math notation.

## Prerequisites
- Manuscript source: `thesis_final/810101504_main.tex`.
- Master results ledger: `results/registry/master_ledger.jsonl`.

## Canonical Audit Files Reference
All refined prose must maintain factual alignment with the six canonical audits:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced for distillation dynamics and Ridge calibration accuracy recovery metrics.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced for low-rank matrix decomposition limits and singular value spectra.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced for 203 unified module performance records and roofline bounds (183.1 FLOP/Byte).
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced for sequence length scaling dynamics and crossover thresholds.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced for architectural invariants and testbed execution parameters.

## Canonical Repository Invariants
Ensure prose strictly adheres to canonical counts:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Rhythm and Cohesion Optimization Techniques

### 1. Dynamic Sentence Cadence Variation
Avoid monotony by alternating between short punchy empirical assertions and compound explanatory sentences:
- *Monotonous*: "FlashAttention-2 achieves 3.12x speedup. Mamba achieves 3.38x speedup. RetNet achieves 2.95x speedup."
- *Refined Rhythm*: "Across the evaluated architectures, native CUDA C++ kernels consistently outperform reference implementations. Notably, FlashAttention-2 attains a $3.12\times$ acceleration \cite{audit03}, whereas recurrent formulations such as Mamba SSM scale to $3.38\times$ by eliminating quadratic matrix materialization."

### 2. Topic and Transition Anchoring
Ensure every paragraph opens with a clear thematic anchor connecting back to the preceding section:
- Contrastive anchors: "In contrast to uncalibrated singular value truncation...", "Despite these theoretical advantages..."
- Empirical anchors: "Corroborating the findings of \cite{audit04}...", "Turning to the 72 compression engines documented in \cite{audit02}..."

### 3. Protection of LaTeX Syntactic Tokens
During prose refactoring, the following syntax must never be altered, renamed, or moved:
- Citation macros: `\cite{audit01}`, `\cite{audit03}`.
- Labels and cross-references: `\label{tab:...}`, `\ref{sec:...}`.
- Monospace identifiers: `\texttt{selective\_ssm}`, `\texttt{hnc\_svd}`.
- Inline math expressions and equations.

---

## Verification
Confirm the polished manuscript satisfies all gates:
1. **Invariant Counts**:
   ```bash
   python3 scripts/utils/verify_readme_counts.py
   # Must verify 131 / 72 / 26
   ```
2. **Audit Reference Check**:
   ```bash
   grep -c "AUDIT_0" thesis_final/810101504_main.tex
   # Must be >= 5
   ```
3. **Compilation Integrity**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must produce 0 fatal errors
   ```

---

## Common Pitfalls
- Accidentally breaking LaTeX macro brackets during prose rephrasing (e.g., editing `\cite{audit01}` into `\cite{audit01`).
- Smoothing away concrete telemetry numbers in favor of generic stylistic phrases.
- Introducing orphan headings or dangling cross-references.
