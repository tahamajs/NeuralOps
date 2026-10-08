# Results Reorganization Plan

## 1. Proposed Target Category Hierarchy

To achieve a clean, maintainable directory structure and separate canonical benchmark evaluations from legacy/ad-hoc experimental artifacts, the following categorization hierarchy is established:

```
results/
├── 01_unified_runs/         # 4-model cross-architecture comparative suites
│   ├── raw/                 # Raw JSON outputs
│   ├── summaries/           # CSV/MD summaries
│   └── plots/               # PNG figures
├── 02_rank_sweeps/          # Multi-rank bottleneck scaling (r = 16 -> 384)
│   ├── raw/
│   ├── summaries/
│   └── plots/
├── 03_zeroshot_evals/       # LM-Eval Harness 5-task evaluations
│   ├── raw/
│   ├── summaries/
│   └── plots/
├── 04_baselines/            # Unmodified foundation model profiles
│   ├── raw/
│   ├── summaries/
│   └── plots/
├── 05_glue_finetuning/      # Multi-seed fine-tuning stability
│   ├── raw/
│   ├── summaries/
│   └── plots/
├── 06_adhoc_experiments/    # Seq scaling, RULER, trajectories
│   ├── seqlen_scaling/
│   ├── ruler_niah/
│   └── misc/
├── 07_misc_tests/           # Master aggregates
│   ├── raw/
│   ├── summaries/
│   └── plots/
├── _archive/                # Historical results and orphaned sweeps
│   ├── 2025/
│   └── pre-2025/
└── _tmp/                    # Scratch space (gitignored)
```

## 2. Reorganization Mapping Sample

| Current Location | Proposed Canonical Path | Classification Rationale |
|:---|:---|:---|
| `results/01_unified_runs/**/*.json` | `results/01_unified_runs/raw/` | Canonical cross-architecture run metrics |
| `results/01_unified_runs/**/*.csv` | `results/01_unified_runs/summaries/` | Tabular summaries |
| `results/02_rank_sweeps/**/*.json` | `results/02_rank_sweeps/raw/` | Rank bottleneck scaling telemetry |
| `results/03_zeroshot_evals/**/*.log` | `results/03_zeroshot_evals/raw/` | Zero-shot LM-Eval logs |
| `results/04_baselines/**/*.json` | `results/04_baselines/raw/` | Uncompressed foundation baseline runs |
| `results/05_glue_finetuning/**/*.json` | `results/05_glue_finetuning/raw/` | GLUE classification evaluation runs |
| `results/06_adhoc_experiments/**/*.png` | `results/06_adhoc_experiments/misc/` | Experimental diagnostic figures |
| `results/verification_2026_09_18/` | `results/07_misc_tests/summaries/` | Verification audit snapshot records |
| `results/adhoc/` | `results/_archive/adhoc/` | Historical exploratory sweeps |

## 3. Orphaned and Incomplete Results Strategy
- 13,606 files lacking canonical module identifiers are categorized as intermediate checkpoints or legacy prototypes.
- Retained safely in `_archive/` without data destruction.
