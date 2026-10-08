# Results Complete Report

## Executive Summary
- Total result files: 39,969
- Total size: ~882 MB
- Modules with results: 217 / 232 (93.9% canonical coverage)
- Coverage matrix: 8.92% across all 231 unique modules × 63 model/dataset combinations
- Models: 55 unique identifiers (7 primary evaluation models)
- Datasets: 12 unique benchmark tasks

## Cleanup Results
- `.bak` removed: 808
- `.err` archived: 6
- Empty removed: 121
- Duplicates removed: 8,886
- Space freed: 59.56 MB

## Coverage Analysis
- Complete modules (covered across multiple architectures): 181
- Partially covered modules (1-2 combinations): 36
- Missing modules (0 benchmark evaluations): 14 (or 15 un-aliased: 14 attention modules, 0 compressor modules)
- Missing combinations: 13,255 zero cells in the 231 × 63 matrix

## Data Quality
- Valid JSON files audited: 17,294 (complete: 8, missing fields: 16,191, non-dict payloads: 1,095)
- Invalid JSON files (parse errors): 34 (mostly trailing commas or multiple concatenated JSON documents)
- Complete records in master summary: 16,199 rows
- Records with valid accuracy: 132 (remainder track loss, perplexity, compression ratio, or qualitative pass status)

## Directory Structure
- Organized into 7 primary categories:
  - `01_unified_runs/`: 4-model cross-architecture comparative suites
  - `02_rank_sweeps/`: Multi-rank bottleneck scaling (r = 16 → 384)
  - `03_zeroshot_evals/`: LM-Eval Harness 5-task evaluations
  - `04_baselines/`: Unmodified foundation model profiles
  - `05_glue_finetuning/`: Multi-seed fine-tuning stability
  - `06_adhoc_experiments/`: Sequence scaling, RULER, trajectories
  - `07_misc_tests/`: Master aggregates
- Archive: 3,029 files (121 MB)
- Errors archive: 6 files (4 KB)
- Master summary: 16,199 rows across 12 tracking metrics

## Recommendations
1. **Benchmark the 14 missing modules**: Conduct unified GLUE runs for missing attention mechanisms (including GPU/Triton kernels `fused_svd_kernel_attention`, `triton_cosformer`, and SSMs `selective_mamba`, `simplessm`).
2. **Fix 34 invalid JSON records**: Clean concatenated JSON dumps and trailing commas in checkpoints.
3. **Delete obsolete archive files**: Remove exploratory distillation curves (`kd_runs/**/*.png`, ~28 MB) and redundant zero-shot logs when disk reclamation is prioritized.
4. **Generate publication tables**: Compile final LaTeX tables from `_LEADERBOARDS.csv` and `_COVERAGE_MATRIX.csv`.

## Next Steps
- GPU benchmark execution for unbenchmarked modules
- PyPI publication
- GitHub release v1.0.0
