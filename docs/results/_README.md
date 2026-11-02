# Results Directory

## Structure

- `01_unified_runs/` — 4-model cross-architecture comparative suites
- `02_rank_sweeps/` — Multi-rank bottleneck scaling (r = 16 → 384)
- `03_zeroshot_evals/` — LM-Eval Harness 5-task evaluations
- `04_baselines/` — Unmodified foundation model profiles
- `05_glue_finetuning/` — Multi-seed fine-tuning stability
- `06_adhoc_experiments/` — Sequence scaling, RULER, trajectories
- `07_misc_tests/` — Master aggregates
- `_archive/` — Historical results
- `_errors_archive/` — Error logs (archived)
- `_tmp/` — Scratch space (gitignored)

## Key Files

- `_MASTER_SUMMARY.csv` — All results in a single table
- `_COVERAGE_MATRIX.csv` — Module × (model, dataset) coverage
- `_LEADERBOARDS.csv` — Top 10 modules per (model, dataset)
- `_LATEX_TABLES/` — Publication-ready LaTeX tables

## Notes

- All module names normalized to canonical registry keys
- All model names cleaned (no paths)
- Duplicate files removed by MD5 hash
- See `docs/RESULTS_FINAL_REPORT.md` for detailed report
