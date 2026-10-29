---
name: paper-arxiv-submission-packager
description: "Validate, package, compile, and gate camera-ready arXiv preprints with verified BibTeX provenance, vector graphics, and zero compilation warnings"
version: "1.0.0"
tags: ["neuralops", "paper", "arxiv", "packaging", "latex", "compilation", "bibtex"]
---

# Skill: Camera-Ready arXiv Submission Packaging and Quality Gating

## When to Use

Use this skill when preparing, compiling, verifying, and packaging manuscripts for official publication on arXiv.org (Primary: `cs.LG`, Secondary: `cs.DC`, `cs.AI`) or top-tier AI and systems conferences (such as ICLR, NeurIPS, MLSys). This workflow guarantees camera-ready quality through rigorous automated checks: pre-generating validated `.bbl` bibliographies, verifying PDF vector font embedding, stripping transient build artifacts (`.aux`, `.log`, `.DS_Store`), asserting zero compilation defects, and maintaining strict alignment with NeuralOps empirical ledgers and zero-fake export laws.

---

## 1. Prerequisites & Target Taxonomies

### 1.1 arXiv Classification & Metadata Specifications
When packaging and uploading the submission to the arXiv electronic preprint server, adhere to the following category mapping:
- **Primary Category**: `cs.LG` (Machine Learning)
- **Secondary Categories**:
  - `cs.DC` (Distributed, Parallel, and Cluster Computing)
  - `cs.AI` (Artificial Intelligence)
- **ACM / IEEE Classification Keywords**: Parallel algorithms, Neural networks, Model compression, Low-rank approximation, Graphics processors (GPUs).
- **Target Repository & Blueprint**: `academic/paper/ARXIV_MASTER_PAPER_BLUEPRINT.md` and submission packages in `academic/paper/iclr2027-submission/` and `academic/paper/neurips2026-submission/`.

### 1.2 Toolchain Requirements
Ensure the following tools and utilities are available in the build environment:
- **LaTeX Engine**: `pdflatex` (TeX Live 2023+ or standard arXiv AutoTeX compatible engine) or `xelatex` (when unicode/system typography is required).
- **Bibliography Engine**: `bibtex` (8-bit or standard 0.99d).
- **Font Inspector**: `pdffonts` (part of `poppler-utils`) to audit embedded Type 1 / TrueType vector fonts.
- **Archive Utility**: POSIX `tar` with gzip compression (`GNU tar` or `bsdtar`).
- **Inspection Tools**: `grep`, `awk`, `find`, `sed`, `python3`.

---

## 2. Canonical Repository & Telemetry Invariants

Every manuscript, table, and bibliography managed by this skill MUST strictly align with the canonical repository invariants defined in `AGENTS.md` and verified in `results/registry/master_ledger.jsonl`:

- **Attention Mechanism Registry**: Exactly **131** attention mechanisms registered in `ATTENTION_REGISTRY`.
- **Model Compression Registry**: Exactly **72** model compression engines registered in `COMPRESSOR_REGISTRY`.
- **Triton Hardware Kernels**: Exactly **26** canonical Triton kernel modules (`neuralops/core/triton_kernels/`), housing 62 unified operators across 4 tiers.
- **Native CUDA C++ Kernels**: Exactly **112** native CUDA C++ kernel packages (`neuralops/core/cuda_kernels/`), housing 114 native implementations.
- **Macro-Telemetry Evaluation Census (RTX 4070 SM 8.9)**:
  - **203** total evaluated logical modules in the master census.
  - **42 Modules `PASS_FAST`**: Statistically significant speedup $>1.05\times$ over PyTorch native SDPA/cuBLAS reference.
  - **161 Modules `PASS_SLOW`**: Correctly routed, verified numerical reference executions ensuring zero regressions.
  - **Defect Count**: Exactly **0 Crashes, 0 Timeouts, 0 Undetected NaN/Inf corruptions**.
- **Hardware Telemetry Baselines**:
  - Uncompressed Baseline Power: $28.74\text{ W}$ mean, $55.3^\circ\text{C}$ steady-state temperature.
  - Compressed Inference: $10.46\text{ W}$ mean (**$-63.6\%$ power reduction**), $48.6^\circ\text{C}$ temperature (**$-6.7^\circ\text{C}$ cooler**).
  - Prefix Caching Throughput: Up to **$4.2\times$** speedup via SGLang RadixAttention integration.

---

## 3. The 4 Zero-Tolerance Quality Gates

Before any arXiv tarball is assembled, the candidate package must clear all four mandatory quality gates. A failure in any single gate halts the release pipeline.

```
═════════════════════════════════════════════════════════════════════════════════════════
                            ARXIV PREPRINT QUALITY GATES
═════════════════════════════════════════════════════════════════════════════════════════
  [Gate 1: Compilation Integrity]
    ├── Exit Code: 0 (Strict)
    ├── Undefined References (\ref, \eqref): EXACTLY 0
    ├── Undefined Citations (\cite, [?]): EXACTLY 0
    └── Overfull \hbox: 0 warnings exceeding 5.0 pt margin threshold
  ─────────────────────────────────────────────────────────────────────────────────────
  [Gate 2: BibTeX Provenance & Hygiene]
    ├── Pre-generated .bbl: Compiled with identical .bib and included in root
    ├── Forensic Audit Links: AUDIT_01 through AUDIT_05 cited with git commit SHAs
    └── Character Escapes: Zero unescaped underscores, percent signs, or ampersands
  ─────────────────────────────────────────────────────────────────────────────────────
  [Gate 3: Vector Graphics & Font Embedding]
    ├── Vector Formats: 100% PDF vector graphics (no compressed JPEG/low-DPI PNG)
    ├── Font Embedding: All fonts Type 1 / TrueType, embedded = yes (pdffonts check)
    └── Zero Raster Loss: Charts, block diagrams, and Pareto curves scale infinitely
  ─────────────────────────────────────────────────────────────────────────────────────
  [Gate 4: Production Serving & Zero-Fake Artifact Compliance]
    ├── Export Contracts: safetensors, GGUF, and ONNX pipelines raise on error
    ├── Serving Engines: Validated vLLM PagedAttention & SGLang RadixAttention APIs
    └── Zero Placeholder Policy: No mock .onnx/.gguf/.trt engine headers
═════════════════════════════════════════════════════════════════════════════════════════
```

### Gate 1: Zero-Warning Compilation Contract
arXiv runs an automated TeX compilation server (AutoTeX). Any unresolved citation or label results in unsightly `[?]` or `??` in the published paper.
- **Exit Code**: Must terminate with code `0`.
- **Undefined References**: `grep -i "LaTeX Warning: Reference" *.log` must return 0 results.
- **Undefined Citations**: `grep -i "LaTeX Warning: Citation" *.log` must return 0 results.
- **Overfull `\hbox`**: All horizontal box overfull warnings greater than `5.0pt` must be eliminated by restructuring prose, tuning hyphenation (`\-`), or wrapping wide math terms.

### Gate 2: BibTeX Hygiene and Forensic Citation Provenance
arXiv **does NOT** execute `bibtex` on their compilation servers; it requires the pre-generated `.bbl` file to be present in the submission archive.
- **Pre-generated `.bbl`**: The file `<main_manuscript>.bbl` must be freshly generated from `<main_manuscript>.bib` and placed in the package root.
- **Forensic Audit Provenance**: Citations to technical reports, audits, and hardware benchmarks must specify forensic tracking fields in their `note` attribute:
  ```bibtex
  @techreport{neuralops_telemetry_2026,
    title = {{NeuralOps Comprehensive Macro Telemetry and Architecture Zoo}},
    author = {Majlesi, Taha and {NeuralOps Contributors}},
    institution = {NeuralOps Open Source Initiative},
    year = {2026},
    note = {Data source: \texttt{results/registry/master\_ledger.jsonl}, git\_commit=\texttt{d14d80e9}, timestamp=\texttt{2026-10-21T18:42:00Z}}
  }
  ```
- **Audit File Reference Keys**:
  - `AUDIT_01_ABLATION_AND_DISTILLATION_EVALUATION.md` $\rightarrow$ `\cite{audit01}`
  - `AUDIT_02_SVD_LOWRANK_AND_RANK_SWEEPS.md` $\rightarrow$ `\cite{audit02}`
  - `AUDIT_03_MACRO_TELEMETRY_AND_HARDWARE_BENCHMARKS.md` $\rightarrow$ `\cite{audit03}`
  - `AUDIT_04_SEQLEN_SCALING_AND_KV_CACHE_MEM.md` $\rightarrow$ `\cite{audit04}`
  - `AUDIT_05_PROVENANCE_INTEGRITY_AND_DISCLOSURE.md` $\rightarrow$ `\cite{audit05}`

### Gate 3: Vector Graphics & Font Embedding Standards
arXiv papers are read on high-resolution displays and printed:
- **Format**: All plots, schematics, and roofline curves must be vector `.pdf` files.
- **Font Embedding**: Every font used in figure labels must be fully embedded. Running `pdffonts figures/<fig>.pdf` must show `emb = yes` for all glyphs.
- **No Type 3 Bitmaps**: Bitmapped Type 3 fonts are strictly prohibited.

### Gate 4: Production Serving & Zero-Fake Artifact Compliance
The paper's claims regarding downstream serving and deployment must faithfully mirror the codebase's strict anti-fake policies:
- **Zero Fake Artifacts**: In accordance with `AGENTS.md §6.1`, the system never generates stub `.onnx`, dummy `.gguf`, or placeholder `.trt` files. Any failure during conversion must explicitly `raise`.
- **vLLM PagedAttention Integration**: Serving models via vLLM must utilize legitimate paged memory layout mapping without unverified padding.
- **SGLang RadixAttention Integration**: Dynamic prefix caching must operate on real token hash radices, yielding valid benchmark speedup records in `master_ledger.jsonl`.

---

## 4. Step-by-Step arXiv Packaging Methodology

Follow this sequence to construct, audit, and compress the arXiv submission package.

### Step 1: Directory Staging and Source Isolation
Stage all source files in an isolated packaging workspace (`build/arxiv_package/` or `academic/paper/arxiv_submission/`).
Use flat relative paths or standardized clean subdirectories:
```
arxiv_submission/
├── neuralops_arxiv.tex         # Main LaTeX source
├── neuralops_arxiv.bbl         # Pre-compiled BibTeX bibliography (MANDATORY)
├── neuralops_arxiv.bib         # Master reference database
├── neurips_2024.sty / custom   # Conference or journal style package (if required)
├── figures/                    # Clean vector PDF figures ONLY
│   ├── pareto_frontier.pdf
│   ├── roofline_rtx4070.pdf
│   └── 3level_fallback.pdf
└── tables/                     # Modular table fragments
    ├── tab_macro_census.tex
    ├── tab_ablation_kd.tex
    └── tab_hardware_power.tex
```

### Step 2: Multi-Pass Compilation & `.bbl` Generation
Run a 4-pass compilation cycle to resolve all citation counters, cross-references, and hyperref links:
```bash
# Set directory context
cd academic/paper/arxiv_submission

# Pass 1: Generate initial .aux and citation keys
pdflatex -interaction=nonstopmode neuralops_arxiv.tex

# Pass 2: Compile bibliography and produce .bbl
bibtex neuralops_arxiv

# Pass 3: Integrate .bbl references into .aux
pdflatex -interaction=nonstopmode neuralops_arxiv.tex

# Pass 4: Finalize page numbers and hyperref cross-links
pdflatex -interaction=nonstopmode neuralops_arxiv.tex
```

### Step 3: Automated LaTeX Warning and Defect Audit
Execute the following automated scan to verify that Gate 1 is satisfied:
```bash
#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="neuralops_arxiv.log"

echo "=== Auditing Compilation Defects in ${LOG_FILE} ==="

# Check 1: Undefined references
UNDEF_REF=$(grep -c "LaTeX Warning: Reference" "$LOG_FILE" || true)
if [ "$UNDEF_REF" -gt 0 ]; then
    echo "FAILED: Found $UNDEF_REF undefined references!"
    grep "LaTeX Warning: Reference" "$LOG_FILE"
    exit 1
fi

# Check 2: Undefined citations
UNDEF_CITE=$(grep -c "LaTeX Warning: Citation" "$LOG_FILE" || true)
if [ "$UNDEF_CITE" -gt 0 ]; then
    echo "FAILED: Found $UNDEF_CITE undefined citations!"
    grep "LaTeX Warning: Citation" "$LOG_FILE"
    exit 1
fi

# Check 3: Overfull hbox > 5pt
OVERFULL=$(grep -E "Overfull \\\\hbox \([0-9]+\.[0-9]+pt too wide\)" "$LOG_FILE" | awk '{
    match($0, /([0-9]+\.[0-9]+)pt too wide/, m);
    if (m[1] > 5.0) print $0;
}' || true)

if [ -n "$OVERFULL" ]; then
    echo "WARNING / DEFECT: Overfull \\hbox exceeding 5pt detected:"
    echo "$OVERFULL"
    exit 1
fi

echo "Gate 1 Passed: 0 undefined references, 0 undefined citations, 0 critical overfull boxes."
```

### Step 4: Font Embedding and Vector Figure Inspection
Inspect all figure PDFs to guarantee full vector font embedding:
```bash
for fig in figures/*.pdf; do
    echo "Inspecting fonts in: $fig"
    # Check for any un-embedded font (emb != yes)
    UNEMBEDDED=$(pdffonts "$fig" | tail -n +3 | awk '$5 != "yes" {print $1, $2, $5}')
    if [ -n "$UNEMBEDDED" ]; then
        echo "ERROR: Unembedded fonts detected in $fig:"
        echo "$UNEMBEDDED"
        exit 1
    fi
done
echo "Gate 3 Passed: All figures contain fully embedded vector fonts."
```

### Step 5: Directory Cleansing (Remove Intermediate Artifacts)
arXiv rejects or warns about stale intermediate build files, local caches, and operating system metadata. Strip them strictly:
```bash
# Clean intermediate LaTeX files (DO NOT REMOVE .bbl or .tex!)
rm -f *.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk *.blg *.dvi

# Remove OS junk files and hidden stores
find . -name ".DS_Store" -delete
find . -name "._*" -delete
find . -name "*.bak" -delete
find . -name "*~" -delete
```

### Step 6: Create the Production arXiv Release Tarball
Archive the directory into a gzip-compressed tarball (`.tar.gz`), keeping paths relative to the root:
```bash
# Package tarball from within the submission directory
tar --exclude='.git' \
    --exclude='.gitignore' \
    --exclude='*.log' \
    --exclude='*.aux' \
    --exclude='*.blg' \
    --exclude='.DS_Store' \
    -czvf neuralops_arxiv_camera_ready.tar.gz *
```

### Step 7: Final End-to-End Extraction & Dry-Run Test
Simulate arXiv's extraction and compilation pipeline in an isolated scratch folder:
```bash
TMP_DIR=$(mktemp -d)
cp neuralops_arxiv_camera_ready.tar.gz "$TMP_DIR/"
cd "$TMP_DIR"
tar -xzvf neuralops_arxiv_camera_ready.tar.gz

# Verify that .bbl exists
if [ ! -f "neuralops_arxiv.bbl" ]; then
    echo "CRITICAL DEFECT: Pre-generated .bbl missing from archive!"
    exit 1
fi

# Run dry-run pdflatex without bibtex (exactly as arXiv does)
pdflatex -interaction=nonstopmode neuralops_arxiv.tex
pdflatex -interaction=nonstopmode neuralops_arxiv.tex

echo "Dry run successful: PDF generated cleanly without running bibtex!"
rm -rf "$TMP_DIR"
```

---

## 5. Comprehensive Quality Gate Checklist

Every release must document this checklist in the packaging report:

```markdown
[ ] 1. Clean Compilation: pdflatex/xelatex terminates with Exit Code 0.
[ ] 2. Zero Unresolved References: Exactly 0 undefined references in .log.
[ ] 3. Zero Unresolved Citations: Exactly 0 citation warnings (no `?` in PDF).
[ ] 4. Overfull \hbox Bounds: Zero overfull boxes exceeding 5.0 pt threshold.
[ ] 5. Pre-Generated .bbl: Present in the archive root, freshly built from .bib.
[ ] 6. Embedded Fonts: All PDF figures show `emb = yes` via pdffonts.
[ ] 7. Clean Directory: 0 .aux, 0 .log, 0 .DS_Store, 0 editor temporary files.
[ ] 8. Canonical Invariants Verified: Exactly 131 attentions, 72 compressors, 26 Triton modules, 203 evaluated modules.
[ ] 9. Verified Attribution & Metadata: Author list, MIT license, GitHub repository URL, and commit SHA included.
[ ] 10. arXiv Categories Defined: Primary `cs.LG`, Secondary `cs.DC`, `cs.AI`.
```

---

## 6. Common Pitfalls & Solutions

1. **Missing Pre-Generated `.bbl` File**:
   - *Symptom*: arXiv AutoTeX compiles the document, but all citations appear as `[?]`.
   - *Cause*: arXiv does not run `bibtex`. If `<manuscript>.bbl` is absent, the bibliography is omitted entirely.
   - *Fix*: Always generate `<manuscript>.bbl` locally and include it in the root of the `.tar.gz`.

2. **Absolute File Paths in `\includegraphics` or `\input`**:
   - *Symptom*: Fatal error `file not found` on arXiv server.
   - *Cause*: Hardcoded paths like `/Users/taha/...` or `C:\...`.
   - *Fix*: Always use clean relative paths: `\includegraphics{figures/pareto_frontier.pdf}`.

3. **Incompatible Hyperref Options**:
   - *Symptom*: `Option clash for package hyperref` or driver warnings.
   - *Fix*: Load `hyperref` last in the preamble, and omit driver-specific parameters like `[dvips]`. Let the LaTeX engine auto-detect the PDF driver.

4. **Stray macOS Resource Forks (`._*`)**:
   - *Symptom*: arXiv unpacker throws warnings or fails to find graphics files because `._figure.pdf` is mistaken for a valid PDF.
   - *Fix*: Run `COPYFILE_DISABLE=1 tar -czvf ...` or `find . -name "._*" -delete` before archiving.

5. **Unescaped LaTeX Reserved Characters in BibTeX Notes**:
   - *Symptom*: Compilation breaks with `Missing $ inserted` inside `.bbl`.
   - *Cause*: Unescaped underscores in URLs or git commit SHAs (e.g. `git_commit=d14d80e9`).
   - *Fix*: Escape as `git\_commit=\texttt{d14d80e9}` or wrap inside `\url{...}`.
