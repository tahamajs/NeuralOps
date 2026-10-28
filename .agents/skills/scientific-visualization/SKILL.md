---
name: scientific-visualization
description: "Author publication-grade scientific figures and multi-panel charts adhering to strict visual guidelines"
version: "1.0.0"
tags: ["neuralops", "visualization", "matplotlib", "figures", "publication"]
---

# Skill: Scientific Visualization and Publication Chart Generation

## When to use
Use this skill when developing, refactoring, or reviewing scientific charts, Pareto frontiers, sequence scaling plots, and distribution histograms for inclusion in NeuralOps publications. It enforces uniform visual aesthetics, vector PDF and high-res PNG export, and strict data provenance.

## Prerequisites
- Manuscript environment: `thesis_final/810101504_main.tex`.
- Master results ledger: `results/registry/master_ledger.jsonl`.
- Python 3.10+ with `matplotlib`, `seaborn`, `numpy`, and `scipy`.

## Canonical Audit Files Reference
All visual artifacts must trace directly back to empirical data in the six canonical audits:

1. `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md`: Data source for KD Regimes 0–5 bar charts and RoBERTa compression decay curves.
2. `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md`: Data source for singular value spectral decay curves and low-rank tradeoff frontiers.
3. `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md`: Data source for the 203-module speedup histogram and 7-family radar charts.
4. `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md`: Data source for log-log sequence scaling curves and memory footprint trajectories.
5. `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md`: Data source for hardware testbed parameters and verification markers.

## Canonical Repository Invariants
Visualizations must represent the canonical repository numbers:
- **131** attention mechanisms in `ATTENTION_REGISTRY`.
- **72** compression engines in `COMPRESSOR_REGISTRY`.
- **26** canonical Triton kernel modules in `neuralops/core/triton_kernels/`.

---

## Visualization Standards and Execution Pipeline

### 1. Style and Typography Configuration
All plotting scripts must configure publication rcParams prior to generating figures:
```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})
```

### 2. Dual-Format Export Standard
Every figure must be saved as both a lossless vector PDF (for LaTeX inclusion) and a 300 DPI raster PNG (for web rendering):
```python
def export_publication_figure(fig, base_path: str):
    fig.savefig(f"{base_path}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
```

### 3. Accessible Color Palettes
Use ColorBrewer or Colorblind-safe palettes (e.g., `tab10`, `Set2`, or `viridis`). Never use indistinguishable red-green pairings for comparative baselines.

### 4. Mandatory Mechanistic Reasoning and Descriptive Accompanying Text
Never treat visualizations as standalone graphics. When embedding any scientific figure in the manuscript:
- The figure MUST be referenced directly (`شکل~\ref{fig:...}`).
- It MUST be accompanied by at least two to three substantive paragraphs of deep physical, algebraic, and microarchitectural explanation:
  - *Hardware root cause*: DRAM memory bus bandwidth saturation, SRAM capacity limits, warp execution stalls, register spilling, and L2 cache overflows (e.g., Chunk GLA 48 MB footprint overflowing the 36 MB L2 cache on Ada Lovelace).
  - *Mathematical foundations*: Singular value energy decay rates, condition number explosion ($\kappa \gg 10^5$), Stiefel manifold Riemannian retraction geometry, and Cholesky whitening pullback.
  - *Ablation attribution*: Explicit causal linkage to the 6 pillars of surgical distillation and the 10 formal theorems.

### 5. English Digits and Labels in All Visuals
All axis numbers, coordinates, legend labels, and data points inside figures and accompanying LaTeX tables MUST use English digits in math mode `$ ... $`. Persian digits in table cells are strictly prohibited to avoid XePersian punctuation and decimal flipping.

---

## Verification
Validate that the visual pipeline satisfies repository requirements:
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
3. **Build Check**:
   ```bash
   cd thesis_final && xelatex -interaction=nonstopmode 810101504_main.tex
   # Must compile cleanly with 0 fatal errors
   ```

---

## Common Pitfalls
- Exporting plots at default 72 DPI, leading to blurry pixelated charts in the printed manuscript.
- Omitting error bars or confidence intervals on stochastic training sweeps.
- Hardcoding synthetic coordinates rather than reading from `results/registry/master_ledger.jsonl`.
