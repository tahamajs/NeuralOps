# 📄 NeuralOps Publications & Academic Artifacts

Welcome to the **NeuralOps Academic Repository** (`SVD_linear_Attention` v1.4.0). This directory serves as the centralized, verified nexus for our primary ICLR 2027 submission manuscript, dual NeurIPS 2026 manuscript, camera-ready LaTeX sources, BibTeX databases, and 247 auto-generated per-module research monographs.

---

## 📂 Architecture and Directory Layout

```
paper/
├── iclr2027-submission/               # PRIMARY SUBMISSION (ICLR 2027)
│   ├── iclr2027_conference.tex        # Master LaTeX manuscript (60 pages, clean)
│   ├── iclr2027_conference.bib        # BibTeX database (87 verified entries)
│   ├── iclr2027_conference.pdf        # Production PDF (1.46 MB, 60 pages)
│   ├── tables/                        # 18 LaTeX table files
│   ├── figures/                       # High-resolution submission figures
│   └── results/                       # Empirical benchmark matrices & CSVs
├── neurips2026-submission/            # DUAL SUBMISSION (NeurIPS 2026)
│   ├── neurips2026_conference.tex     # Master LaTeX manuscript (60 pages, clean)
│   ├── iclr2027_conference.bib        # BibTeX database (87 verified entries)
│   ├── neurips2026_conference.pdf     # Production PDF (1.43 MB, 60 pages)
│   ├── tables/                        # 18 LaTeX table files
│   └── figures/                       # High-resolution submission figures
├── build_paper/                       # CANONICAL MONOGRAPHS (247 Modules)
│   ├── <module_name>/                 # 19 standalone structured sections
│   │   ├── 00_meta.md ... E_ethics.md # Markdown sections
│   │   ├── 00_meta.tex ... E_ethics.tex
│   │   ├── paper_full.md              # Unified Markdown monograph
│   │   ├── paper_full.tex             # Unified LaTeX monograph
│   │   └── main.tex                   # Compilable document entrypoint
│   └── ...
├── per_module -> build_paper          # Symlink alias for per-module monographs
├── scripts/                           # Monograph generation & section builders
│   ├── build_paper.py                 # Master generator CLI
│   ├── enhanced_sections.py           # Universal section synthesis engine
│   └── enrich_module_paper.py         # Empirical metadata injector
├── figures/                           # Master vector graphic assets (PDF / SVG / PNG)
├── tables/                            # Standalone supplementary LaTeX tables
├── reports/                           # 66 Technical monographs and theoretical audits
├── audit/                             # Screening logs, verification matrices, and caches
├── iclr2027-abstract/                 # Extended abstract and poster draft materials
└── README.md                          # Repository layout and instructions (this file)
```

---

## 📊 Summary of Directory Roles

| Directory | Contents | Primary Purpose |
|:---|:---|:---|
| `iclr2027-submission/` | Main LaTeX + BibTeX + Tables + Figures | Canonical camera-ready ICLR 2027 manuscript (60 pages, `iclr2027_conference.sty`) |
| `neurips2026-submission/` | Main LaTeX + BibTeX + Tables + Figures | Canonical camera-ready NeurIPS 2026 manuscript (60 pages, `neurips_2026.sty`) |
| `build_paper/` | 247 directories, ~10,000 files | Comprehensive 19-section research papers for all modules |
| `per_module/` | Symlink pointing to `build_paper/` | Convenient alias for automated toolchains |
| `scripts/` | Python CLI and enrichment tooling | Generation and validation pipeline for module papers |
| `figures/` | High-resolution publication graphics | Vector and raster master assets for submissions |
| `reports/` | 66 Markdown monographs | Theoretical whitepapers and diagnostic reports |
| `audit/` | 14 screening files and logs | Complete reproducibility, literature, and quality audits |

---

## 🛠️ Reproduction & Compilation Protocols

### 1. Compiling the Primary ICLR 2027 & NeurIPS 2026 Papers

Execute the standard 4-pass compilation pipeline locally:

**ICLR 2027 Compilation:**
```bash
cd paper/iclr2027-submission
pdflatex -interaction=nonstopmode iclr2027_conference.tex
bibtex iclr2027_conference
pdflatex -interaction=nonstopmode iclr2027_conference.tex
pdflatex -interaction=nonstopmode iclr2027_conference.tex
```
*Expected Output*: `iclr2027_conference.pdf` (60 pages, 0 undefined citations, 0 undefined references, 0 missing figures).

**NeurIPS 2026 Compilation:**
```bash
cd paper/neurips2026-submission
pdflatex -interaction=nonstopmode neurips2026_conference.tex
bibtex neurips2026_conference
pdflatex -interaction=nonstopmode neurips2026_conference.tex
pdflatex -interaction=nonstopmode neurips2026_conference.tex
```
*Expected Output*: `neurips2026_conference.pdf` (60 pages, 0 undefined citations, 0 undefined references, 0 missing figures).

---

### 2. Generating Per-Module Monographs

All per-module monographs can be built or regenerated using `paper/scripts/build_paper.py`:

**Build a single module monograph:**
```bash
python paper/scripts/build_paper.py --module adasvd_compressor
```

**Build all 247 module monographs:**
```bash
python paper/scripts/build_paper.py --all
```

*Each monograph generates `paper_full.md`, `paper_full.tex`, and `main.tex` across 19 standard academic sections.*

---

## 🔬 Canonical Verification Invariants

NeuralOps maintains strict architectural invariants across the repository:
- **131 Attention Mechanisms**: Verified in `neuralops.core.registry.ATTENTION_REGISTRY`
- **72 Compression Engines**: Verified in `neuralops.core.registry.COMPRESSOR_REGISTRY`
- **26 Custom Triton GPU Kernels**: Verified in `neuralops/core/triton_kernels/`

Verify locally or remotely via:
```bash
python scripts/utils/verify_readme_counts.py
```

---

## 📈 Verified Empirical Ledger & SOTA Gap Closures

The 60-page comprehensive manuscripts integrate results from physical hardware execution sweeps on NVIDIA RTX 4070 SM 8.9:
- **Five SOTA Gap Closures**: Fully benchmarked and mathematically proven in [`results/SOTA_GAP_CLOSURE_REPORT.md`](../results/SOTA_GAP_CLOSURE_REPORT.md):
  1. *SVD-LLM v2 Truncation Optimality* (Eckart-Young-Mirsky via Cholesky whitening)
  2. *ASVD KV-Cache & Streaming Latent Forward* ($\mathcal{O}(L \cdot r)$ recurrent state compression)
  3. *FP8 e4m3fn Bounding* ($< 5\%$ quantization relative error)
  4. *DSA Learned Block-Sparse Routing* ($\mathcal{O}(T \cdot k \cdot B \cdot d)$ with $\ge 95\%$ dense gradient alignment)
  5. *REAP Distributed MoE Pruning* (Unbiased router statistics aggregation)
- **Repository Provenance**: Verified on NeuralOps v1.4.0 (`fdfa48dfb50bf4c97dd3e113a8e822cb91e16a96`).
- **Physical Benchmarks Ledger**: Ingested in `paper/iclr2027-submission/results/` and `paper/neurips2026-submission/tables/`.

