---
name: scientific-writing
description: "Calibrate academic formality, rigorous hedging, and claim-evidence alignment for scientific manuscripts"
version: "1.0.0"
tags: ["neuralops", "writing", "academic", "scientific", "prose"]
---

# Skill: Scientific Writing and Academic Formality Calibration

## When to use
Use this skill when drafting, revising, or refining prose across thesis chapters and conference manuscripts. It enforces rigorous academic style, eliminates speculative or hyperbolic claims, calibrates epistemic hedging, and ensures that every sentence is backed by verified data from the NeuralOps audit files.

## Prerequisites
- Target document: `thesis_final/810101504_main.tex`.
- Master results ledger: `results/registry/master_ledger.jsonl`.

## Canonical Audit Files Reference
All statements of fact, capability assertions, and comparative analyses must trace back to the canonical audits:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Sourced for knowledge distillation claims, RoBERTa accuracy recovery ($73.62\%$), and Ridge regularization bounds.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Sourced for low-rank matrix factorization bounds, Eckart-Young-Mirsky optimality, and singular value spectral decay.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Sourced for 203 unified module latencies, 131 attention operator performance benchmarks, and hardware roofline limits.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Sourced for asymptotic complexity comparisons and KV cache memory scaling.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Sourced for architectural invariants and testbed execution parameters.

## Canonical Repository Invariants
Writing must strictly reflect repository invariants:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Writing Principles and Calibration Guidelines

### 1. Epistemic Hedging and Evidence Strength
Calibrate claim certainty to the empirical findings in `results/registry/master_ledger.jsonl`:
- **Strong Evidence (Direct measurement)**: Use active verbs (`demonstrates`, `proves`, `achieves`, `measures`).
  *Example*: "Profiling on RTX 4070 SM 8.9 proves that FlashAttention-2 achieves $3.12\times$ speedup over the PyTorch baseline \cite{audit03}."
- **Moderate Evidence (Observed trend)**: Use calibrated markers (`indicates`, `suggests`, `corresponds to`).
  *Example*: "The singular value spectral decay curve indicates that the first 64 components capture $94.2\%$ of total energy \cite{audit02}."
- **Speculative Claims**: Strictly banned. Never claim an architecture "will scale to trillion-parameter models" without empirical validation.

### 2. Elimination of AI Conversational Boilerplate
Strip away colloquial conversational filler and generic buzzwords:
- Banned: "It is crucial to note that", "delve into", "testament to", "game-changing".
- Preferred: "Empirical telemetry indicates", "Table 4.1 details", "As demonstrated in \cite{audit01}".

### 3. Rigorous Notation, Unit Typesetting, and Table English Digits
- Use standard math mode for symbols and italicized dimensions: $N \in [128, 8192]$, $B=16$, $d=64$.
- Use explicit non-breaking thin spaces between numbers and SI units: `2.4\,\text{GB}`, `0.421\,\text{ms}`, `255.8\,\text{GB/s}`.
- **Mandatory English Math-Mode Digits in Tables**: All digits, percentages, speedups, and dimensions in tables MUST be formatted in English math mode `$ ... $` (e.g. `$94.50\%$`, `$1.74\times$`, `$0.0089\text{ ms}$`, `$34.80\%$`, `$r=32$`). Persian digits in table cells are strictly prohibited to prevent decimal point inversion and sign flipping in XePersian.
- **Zero Table Overflow & Widen Layout**: Ensure tables fit within page margins (`Zero Overfull \hbox`) by using `tabularx` with `X` columns, `\small` or `\footnotesize`, and adjusted `\tabcolsep`.

### 4. Mandatory Visual Analytics & Deep Mechanistic Reasoning
- **Dense Scientific Figures**: Manuscripts must feature high-density visual analytics (Pareto frontiers, rooflines, scaling plots, speedup histograms, radar charts, loss/perplexity landscapes, memory allocation heatmaps) from `academic/thesis/figures/` and `academic/thesis/img/`.
- **Deep Mechanistic Reasoning**: Every figure and table MUST be accompanied by at least two to three substantive paragraphs of deep physical, algebraic, and microarchitectural reasoning:
  - *Hardware physics*: SRAM/DRAM bandwidth, register pressure (e.g. 80 registers/thread reducing warp occupancy in FP8), cache thrashing (Chunk GLA 48 MB footprint overflowing 36 MB L2 cache), memory bus transaction coalescing.
  - *Mathematical geometry*: Stiefel manifold Riemannian retraction, condition number explosion ($\kappa(\mathbf{S}_X) \gg 10^5$), Cholesky whitening pullback, discrete recovery derivative $\Delta(r)$, rank collapse bounds.
  - *Causal attribution*: Never state raw empirical numbers without explaining the underlying physical and mathematical causes.

---

## Verification
Validate that the prose complies with quality gates:
1. **Canonical Registry Counts**:
   ```bash
   python3 scripts/utils/verify_readme_counts.py
   # Must return 131 / 72 / 26
   ```
2. **Audit Citations**:
   ```bash
   grep -c "AUDIT_0" thesis_final/810101504_main.tex
   # Must be >= 5
   ```
3. **Manuscript Build**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must exit with 0 errors
   ```

---

## Common Pitfalls
- Stating universal generalizations ("linear attention is always faster than softmax") without citing sequence crossover bounds ($N \approx 2048$ from `AUDIT_04`).
- Overusing passive voice in methodology sections where active procedural verbs are clearer.
- Substituting vague qualitative adjectives ("significant improvement") for exact quantitative deltas ("$+21.85\%$ tail energy preservation").
