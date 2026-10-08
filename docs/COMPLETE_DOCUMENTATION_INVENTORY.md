# NEURALOPS: COMPLETE DOCUMENTATION AND PAPERS INVENTORY
Generated on: 2026-10-08 | Scope: Repository-wide documentation, manuscripts, and reports

## 1. EXECUTIVE SUMMARY

- **Total Manuscripts / Papers**: **213**
  - 1 Flagship Conference Paper: *Toward Linear Attention: A Comprehensive Comparison of Over 200 Attention Mechanisms* (`paper/iclr2027-submission/iclr2027_conference.tex`)
  - 1 Conference Abstract Package: `paper/iclr2027-abstract/abstract.tex`
  - 211 Module-Specific Papers: Complete publication packages under `paper/build_paper/<module>/` with dedicated math, background, benchmarks, and compiled PDFs.
- **Total Documentation Files (`docs/`)**: **377 files** (78 root guides/audits, 6 blog compendia, 242 module deep-dives, 16 dashboard specs, 11 UI manuals, 5 tutorials).
- **Total Markdown Files (.md)**: **5,950** repository-wide.
- **Total LaTeX Files (.tex)**: **4,639** files.
- **Total Compiled PDFs**: **514** compiled documents.
- **Total Bibliographies (.bib)**: **5** database files.
- **Documentation Completeness**: **100% complete** across core docs (0 unresolved TODO/FIXME markers).
- **Broken Links**: 1,177 internal links (predominantly absolute IDE `file:///Users/tahamajs/...` paths created during development audits).
- **Version Status**: Aligned at `v1.1.0` in `pyproject.toml`, `neuralops/__init__.py`, and `CHANGELOG.md`; `CITATION.cff` flags `1.0.0`.

## 2. REPOSITORY DOCUMENTATION DIRECTORY STRUCTURE

```text
Project/
├── docs/                                    # Framework documentation, blogs, and audits
│   ├── Attention_Visualization_Dashboard/   # UI/UX specification and manual
│   ├── Mapping_Apply_the_Attention.../      # Component mapping & interactive studios
│   ├── blog/                                # Masterclass Deep-Dive Series (Tracks 1-5)
│   ├── content/                             # Deep architectural analysis
│   │   ├── attentions/                      # 138 individual attention mechanism guides
│   │   ├── compressions/                    # 90 individual compression engine guides
│   │   └── framework/                       # Next-gen innovations & synthesis compendia
│   ├── tutorials/                           # Step-by-step developer tutorials
│   └── visualizer/                          # 3D interactive visualizer React/ThreeJS suite
├── paper/                                   # Research publications & LaTeX manuscripts
│   ├── iclr2027-submission/                 # Main flagship conference submission
│   │   ├── iclr2027_conference.tex         # Complete conference source
│   │   ├── iclr2027_conference.bib         # Conference bibliography
│   │   └── iclr2027_conference.pdf         # Compiled publication PDF
│   ├── iclr2027-abstract/                   # Conference abstract & data summary
│   ├── build_paper/                         # 211 modular publication packages
│   │   └── <module>/                        # 00_meta to 13_references, tex & pdf
│   ├── audit/                               # Paper verification, citation & HTTP audits
│   ├── reports/                             # Evaluation reports & LaTeX bibliography
│   └── template/                            # Standard ICLR conference templates
├── audit/                                   # Codebase architectural and numeric audit logs
└── root (.md files)                         # Top-level user-facing documentation
```

## 3. COMPREHENSIVE PAPER INVENTORY

### 3.1 Primary Conference Manuscripts

| File | Title | Venue | Format | Size | Lines |
|---|---|---|---|---|---|
| `paper/iclr2027-submission/iclr2027_conference.tex` | Toward Linear Attention: A Comprehensive Comparison of Over 200 Attention Mechanisms | ICLR 2027 | LaTeX | 43 KB | 632 |
| `paper/iclr2027-submission/iclr2027_conference.pdf` | Toward Linear Attention: A Comprehensive Comparison of Over 200 Attention Mechanisms | ICLR 2027 | PDF | 512 KB | - |
| `paper/iclr2027-abstract/abstract.tex` | Toward Linear Attention (Abstract Submission) | ICLR 2027 | LaTeX | 3.2 KB | 78 |
| `paper/iclr2027-abstract/abstract.pdf` | Toward Linear Attention (Abstract Submission) | ICLR 2027 | PDF | 114 KB | - |

### 3.2 Modular Paper Corpus (`paper/build_paper/`)

NeuralOps maintains **211 standalone publication packages** corresponding to each evaluated module, each containing:
- `00_meta.md` (Title, Authors, Abstract, Keywords)
- `01_introduction.tex` & `02_related_work.tex`
- `03_background.tex` & `04_methodology.tex`
- `05_architecture.tex` & `06_empirical_results.tex`
- `07_ablation_studies.tex` & `08_comparative_analysis.tex`
- `09_practical_implications.tex` & `10_limitations.md`
- `11_conclusion.tex` & `12_reproducibility.md`
- `13_references.md` / `.bib`
- `main.pdf` & `main_final.pdf` (Fully compiled camera-ready paper)

## 4. DOCUMENTATION INVENTORY (`docs/`)

| Category | Directory | File Count | Description |
|---|---|---|---|
| Root Guides & Audits | `docs/` | 78 | System architecture, CLI specs, audit roadmaps, and verification logs. |
| Masterclass Blog Series | `docs/blog/` | 6 | 5 thematic publication tracks + comprehensive compendium (Tracks 1-5). |
| Attention Deep Dives | `docs/content/attentions/` | 138 | Mathematical formulation, complexity, and usage for attention layers. |
| Compression Deep Dives | `docs/content/compressions/` | 90 | Theory, calibration protocols, and trade-offs for compression engines. |
| Interactive Studio Specs | `docs/Mapping_Apply.../` | 16 | Specification for residual stream, RoPE, loss, and LoRA studios. |
| Visualizer UI Manuals | `docs/Attention_Visual.../` | 11 | Complete UI design system, interaction modes, and keyboard shortcuts. |
| Tutorials | `docs/tutorials/` | 5 | Getting started guides, custom module authoring, and benchmark runners. |
| Root Documents | Repository Root | 6 | `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `MASTER_PROJECT_RESULTS.md`, `CONSOLIDATED_MASTER_AUDIT_REPORT.md`. |

## 5. REPOSITORY CONSISTENCY AUDIT

| Metric | Declared / Target | Actual Discovered | Status |
|---|---|---|---|
| Framework Version | 1.1.0 | `1.1.0` (pyproject.toml, neuralops, CHANGELOG) | ✅ Consistent (CITATION.cff has 1.0.0) |
| Attention Modules | 159 | 159 Registered in ATTENTION_REGISTRY | ✅ Verified Exact |
| Compression Engines | 73 | 73 Registered in COMPRESSOR_REGISTRY | ✅ Verified Exact |
| Triton Kernels | 16 | 16 Fused Kernels in `neuralops.core.triton_kernels` | ✅ Verified Exact |
| Documentation Completeness | 100% | 0 TODO / FIXME markers in active docs | ✅ Complete |
| Broken Internal Links | 0 | 1,177 Broken Links (Legacy absolute `file:///` paths) | ⚠️ Flagged for cleanup |

## 6. RECOMMENDATIONS FOR DOCUMENTATION HYGIENE

1. **Update `CITATION.cff`**: Bump declared version from `1.0.0` to `1.1.0` to maintain perfect version parity with `pyproject.toml`.
2. **Normalize Link Paths**: Convert absolute IDE file URLs (`file:///Users/tahamajs/Documents/uni/Project/...`) in audit markdown reports to relative markdown links.
3. **Archive Stale Audit Runs**: Consolidate redundant one-off audit logs in `audit/` into a single timestamped `audit/_archive/` directory to improve repository traversal speed.