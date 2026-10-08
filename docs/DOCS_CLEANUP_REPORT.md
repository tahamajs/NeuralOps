# Documentation Cleanup Report (Part A)

**Date**: 2026-10-08  
**Scope**: Repository-wide documentation, link validation, bloat investigation, and metadata synchronization.

---

## 1. Summary of Actions & Findings

### Task A.1 — Documentation Bloat Investigation
- **Total Markdown (.md) files**: 5,950 repository-wide (4,299 located in `paper/build_paper/`, 1,009 in `results/`, 377 in `docs/`).
- **Total LaTeX (.tex) files**: 4,639 across publication and thesis packages.
- **Total Compiled PDFs**: 514 across publication packages.
- **Top 3 Largest Markdown files**:
  1. `docs/content/bad_results.md` (10,469 lines)
  2. `paper/audit/_AUDIT_PER_PAPER.md` (5,582 lines)
  3. `audit/FULL_TEMPLATE_ARTIFACT_AUDIT_REPORT.md` (4,429 lines)
- **Top Directory Sizes**:
  - `paper/`: 1.7 GB (dominated by `build_paper/` containing 211 module packages + PDFs)
  - `results/`: 819 MB
  - `thesis_final/`: 493 MB
  - `neuralops/`: 232 MB
  - `docs/`: 73 MB

### Task A.2 — Version Synchronization
- Synchronized `CITATION.cff` from `1.0.0` to `1.1.0`.
- Verified parity across all metadata files:
  - `pyproject.toml`: `1.1.0`
  - `neuralops/__init__.py`: `1.1.0`
  - `CHANGELOG.md`: `1.1.0`
  - `CITATION.cff`: `1.1.0`

### Task A.3 — Broken Local `file:///` Links
- Identified 8 documentation files containing absolute IDE paths (`file:///Users/tahamajs/Documents/uni/Project/...`).
- Converted all instances to portable relative links (`../`). Verified 0 lingering instances.

### Task A.4 — Audit Directory Analysis
- `audit/` contains 46 markdown files and 97 directory entries.
- Recommended future step: archive one-off audit logs prior to release into `audit/_archive/2026_10/`.

### Task A.5 — Internal Link Audit
- Surveyed all internal markdown links:
  - 2,000 broken links identified, largely referencing historical paths from older development trees (e.g. `Codes/framework/...`).
  - Active links within `docs/` and `README.md` verified functional.

### Task A.6 — `paper/build_paper/` Structure
- 219 module folders containing 12,527 files.
- Average 48–50 files per module (TeX, metadata, citations, and dual PDFs).
- Total disk size: 1.7 GB.

---

## 2. Recommendations for Future Maintenance
1. Move historical `Codes/framework/...` references in archived experiment logs to relative historical pointers.
2. Consider compressing or git-LFS tracking pre-compiled PDF manuscripts in `paper/build_paper/` to streamline repository cloning.
