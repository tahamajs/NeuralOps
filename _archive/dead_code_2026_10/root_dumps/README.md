# Archived Root Dumps & Obsolete Artifacts (October 2026)

This directory preserves obsolete standalone log dumps and screening summaries moved from the repository root:
- `run_quality_all_screening_summary.csv` (Preserved permanently in `results/run_quality_all_screening_summary.csv` and subdirectories)
- `benchmark_accuracy_and_ppl.log` (Obsolete benchmark execution log)
- `benchmark_accuracy_and_ppl_clean.log` (Obsolete benchmark execution log)
- `grading.log` (Local test runner grading artifact)
- `smoke_svd.log` / `smoke_svd.err` (0-byte smoke execution log artifacts)

No Python source code, kernel files, or test suites were modified or removed.

### Loose Non-Module Files & Temporary Bundles Archive
- `.bundle` artifacts (15 patch bundles: `fix_all.bundle`, `fix_baseline.bundle`, `update_sync*.bundle`, `wikitext_fix.bundle`, etc.) safely consolidated.
- Stray auxiliary LaTeX build outputs (`master_thesis.aux`, `*.lof`, `*.lot`, `*.out`) archived from root.
- Removed accidentally created local truncation filename artifact (`bhkv, k_t.float(), v_t.float())`).
- Verified canonical root entrypoints (`__init__.py`, `__main__.py`, `api.py`, `main.py`, `py.typed`, `recipes.py`, `pyproject.toml`, `README.md`).
