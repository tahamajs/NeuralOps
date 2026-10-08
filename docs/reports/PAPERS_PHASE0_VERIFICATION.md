# Phase 0 Verification Report: Deep Audit of Papers

**Date:** 2026-10-09  
**Branch:** `ablation`  
**Target:** `paper/build_paper/` research packages  

---

## 1. Paper Inventory (Step 0.1)

- **Total directories in `paper/build_paper/`**: 219
- **Non-infrastructure paper folders**: 219
- **Infrastructure files/folders**:
  - `paper/build_paper/_PROGRESS_TRACKER.md` (tracking manifest)
  - `paper/templates/` (`_shared`, `_TEMPLATE`) cleanly separated

---

## 2. Duplicate Detection (Step 0.2)

### Pairwise Identical Content Clusters Found (7 Pairs / 14 Folders):
Exact file-level duplication identified between legacy CamelCase/stub folders and canonical snake_case folders:
1. `['AdaptiveSVD', 'adaptive_svd_attention']`
2. `['AdaptiveSpanAttention', 'adaptive_span_attention']`
3. `['AdaptiveSparseAttention', 'adaptive_sparse_attention']`
4. `['Galerkin', 'galerkin_attention']`
5. `['SOAFusedAttention', 'soa_fused_module']`
6. `['aa_svd_compressor', 'aasvd']`
7. `['flash3', 'flash_attention3']`

*Assessment:* These 7 legacy pairs are literal duplicates created when populating stubs. In Phase 1, they must either be merged/aliased or given dedicated, distinct technical content.

---

## 3. Word Count & Content Depth (Step 0.3)

All 219 papers have extensive content depth:
- Lowest word count among papers: `padre_attention` (43 files, 24,798 words)
- Highest word count: `cur_moe` (42 files, 31,926 words)
- Average words per paper package: ~29,500 words across all `.md` and `.tex` sections.
- Zero empty or short stub folders remain.

---

## 4. Placeholder Detection (Step 0.4)

- **Canonical Section Files (`0*.md`, `0*.tex`, `1*.md`, `1*.tex`)**: **0 placeholder occurrences**. No `TODO`, `FIXME`, `Lorem ipsum`, or `<FILL>` tokens exist in any canonical paper sections.
- **Checklist/Tracker Artifacts**: 82 occurrences confined exclusively to internal drafting logs (`contents.md`, `checklist.md`, `Template.md`).

---

## 5. LaTeX Compilation Sample (Step 0.5)

Tested 5 representative sample packages with `pdflatex`:
1. `adaptive_svd_attention`: Math syntax warning (`! Missing $ inserted`)
2. `gated_linear_attention`: Math mode delimiter issues in inline equations
3. `cosformer`: Command `\item` invalid in math mode
4. `performer_attention`: Missing delimiter / bracket closure
5. `mamba_block`: Directory is named `mamba` or `svd_mamba` in `build_paper/`

*Assessment:* While pre-compiled PDFs (`main.pdf`, `main_final.pdf`) exist in all 219 folders, several raw LaTeX source packages require formula and delimiter hardening to compile cleanly from scratch.

---

## 6. Version State (Step 0.6)

All manifests are 100% synchronized to version **1.2.0**:
- `CITATION.cff`: `version: 1.2.0`
- `pyproject.toml`: `version = "1.2.0"`
- `neuralops/__init__.py`: `__version__ = "1.2.0"`
- `CHANGELOG.md`: `## [1.2.0] - 2026-10-09`

---

## 7. Registry Mapping Discrepancies (Step 0.7)

- **76 folders** use names that do not match exact canonical registry keys (e.g. `AdaptiveSVD` vs `adaptive_svd_attention`, `Galerkin` vs `galerkin_attention`, `SOAFusedAttention` vs `soa_fused_module`).
- **90 registry keys** lack a 1:1 identical directory name (they map to shortened aliases or shared architectural family folders).

---

## Conclusion & Proposed Phase 1 Actions

1. Deduplicate the 7 duplicate pairs (`AdaptiveSVD`, `AdaptiveSpanAttention`, `AdaptiveSparseAttention`, `Galerkin`, `SOAFusedAttention`, `aa_svd_compressor`, `flash_attention3`) by consolidating them with their canonical packages.
2. Resolve LaTeX syntax glitches identified in sample compilations.
3. Clean up folder-to-registry name mappings.
