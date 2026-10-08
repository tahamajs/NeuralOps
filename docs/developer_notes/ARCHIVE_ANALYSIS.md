# Archive Analysis

Detailed audit of the archived results, historical logs, and scratch artifacts in `results/_archive/` and `results/_errors_archive/`.

## 1. Directory Overview

| Directory | File Count | Size (MB) | Primary File Types | Date Range |
|:---|---:|---:|:---|:---|
| `results/_archive/audit/` | 6 | 0.08 MB | `.json`, `.log`, `.csv`, `.txt` | 2026-10-04 to 2026-10-05 |
| `results/_archive/ablation/` | 13 | 0.05 MB | `.log` | 2026-10-08 |
| `results/_archive/adhoc/` | 1,006 | 38.73 MB | `.json` (zeroshot parallel runs) | 2026-10-04 |
| `results/_archive/kd_runs/` | 1,924 | 79.71 MB | `.json`, `.png`, `.txt`, `.csv` | 2026-10-05 to 2026-10-08 |
| `results/_archive/verification_2026_09_18/` | 8 | 0.00 MB | `.csv`, `.json` | 2026-10-08 |
| `results/_archive/old_tests/` | 72 | 0.17 MB | `.json`, `.csv`, `.md` | 2026-09-27 to 2026-10-04 |
| `results/_errors_archive/` | 6 | 0.00 MB | `.err` error logs | Historical |
| `results/_tmp/` | 0 | 0.00 MB | *(Directory absent / empty)* | N/A |
| **Total Archived** | **3,035** | **~121.0 MB** | — | — |

---

## 2. Archive Contents Analysis

1. **`kd_runs/` (1,924 files, 79.71 MB)**:
   - Contains intermediate distillation plots (`.png` files, 666 items), loss trajectory logs, and checkpoints for knowledge distillation runs.
   - Preserves training progression curves and raw outputs from early KD experiments.

2. **`adhoc/zeroshot_parallel/` (1,006 files, 38.73 MB)**:
   - Historical parallel zero-shot evaluations conducted prior to the structured consolidation into `03_zeroshot_evals/`.

3. **`old_tests/` (72 files, 0.17 MB)**:
   - Early pilot validation runs and legacy experiment summaries.

4. **`_errors_archive/` (6 files, 4.0 KB)**:
   - Stored crash logs and traceback dumps from unhandled exceptions during pipeline debugging.

---

## 3. Recommended Deletions (Pending User Approval)

The following groups of files are identified as candidates for permanent deletion if disk savings are desired:

1. **Intermediate KD Plot Artifacts (`kd_runs/**/*.png`)**:
   - 666 image files (~28 MB). These are visual artifacts from exploratory KD trials whose quantitative metrics are already recorded.
2. **Ad-hoc Duplicate Evaluator Dumps**:
   - Outdated runs from `_archive/adhoc/zeroshot_parallel/` whose canonical summaries are fully subsumed in `03_zeroshot_evals/`.
3. **Empty or Truncated Error Dumps**:
   - 0-byte or uninformative error files in `_errors_archive/`.

*Note: Per the operational mandate, no files have been deleted during this analysis.*
