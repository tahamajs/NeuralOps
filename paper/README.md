# Scientific Papers & Publications Directory

This directory houses all manuscript artifacts, conference submissions, abstracts, and per-module short papers for our large-scale empirical study on linear attention, low-rank factorization, and vector quantization across 200+ transformer mechanisms.

## Directory Structure

```text
paper/
├── iclr2027-submission/        # Primary ICLR 2027 Main Conference Paper (Double-blind submission)
│   ├── iclr2027_conference.tex # Master LaTeX manuscript (strictly within 9-page limit)
│   ├── iclr2027_conference.pdf # Full compiled PDF (14 pages: 8.5 main + refs + appendices)
│   ├── figures/                # High-resolution benchmark figures and Pareto trade-off curves
│   ├── tables/                 # Modular LaTeX table definitions (hyperparameters, sweeps, Pareto)
│   └── results/                # Canonical summary CSV datasets (SST-2, MNLI, AG News)
│
├── iclr2027-abstract/          # Official Abstract Registration Artifacts
│   ├── abstract.tex            # Canonical 246-word abstract source
│   ├── abstract.pdf            # Compiled abstract artifact
│   └── final/                  # Final immutable submission release
│
├── template/                   # Official ICLR 2027 LaTeX Style Templates
│   └── iclr2027/               # Base style files (iclr2027_conference.sty, bst, math_commands)
│
├── references_arxiv/           # Archived Primary Literature Reference PDFs
│   └── (Foundational arXiv papers: Vaswani et al., Sanh et al., Linformer, Performer, etc.)
│
├── PerModulePaper/             # Collection of 217 Per-Module ICLR 2027 Workshop Papers
│   ├── _TEMPLATE/              # Master scaffolding blueprint
│   ├── a3/                     # Complete ICLR 2027 workshop paper for A3
│   ├── kmeans_quantize/        # Complete ICLR 2027 workshop paper for K-Means Quantization
│   ├── AdaptiveSVD/            # Complete ICLR 2027 workshop paper for Adaptive SVD
│   ├── ...                     # 217 fully compiled modules in exact ICLR double-blind style
│   └── README.md               # Master directory and hyperlinked PDF catalog
│
└── README.md                   # This directory guide
```

## Quick Links to Key Manuscripts

- **Main ICLR 2027 Paper:** [`iclr2027-submission/iclr2027_conference.pdf`](iclr2027-submission/iclr2027_conference.pdf)
- **Official Abstract:** [`iclr2027-abstract/abstract.pdf`](iclr2027-abstract/abstract.pdf)
- **Per-Module Paper Catalog:** [`PerModulePaper/README.md`](PerModulePaper/README.md)
