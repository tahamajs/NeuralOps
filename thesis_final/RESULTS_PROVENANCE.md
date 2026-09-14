# Thesis results provenance

This file records which result files are safe to cite in the thesis. Numbers must not be
copied between runs, model families, or datasets without stating that change.

## Current paper-run snapshot

The reproducible paper-run table is `../results/final_paper_run/summary.csv`.
It contains 384 rows for 203 unique module names, with 333 `PASS`, 49 `FAIL`, and 2
`TIMEOUT` rows. Its evaluated model family is `DISTILBERT`, on `SST2` and `MRPC`.

For the two directly comparable rows used in the thesis:

| module | dataset | accuracy | F1 | peak VRAM (MB) | throughput (tokens/s) | status |
|---|---|---:|---:|---:|---:|---|
| `full` | SST2 | 90.25 | 0.9044 | 1338.6 | 61153.2 | PASS |
| `fusion_attn` | SST2 | 90.02 | 0.9037 | 1419.9 | 46835.6 | PASS |
| `full` | MRPC | 90.02 | 0.9028 | 1336.7 | 61072.3 | PASS |
| `fusion_attn` | MRPC | 90.02 | 0.9037 | 1421.9 | 45752.0 | PASS |

These rows do not support the older claims that Fusion Attention achieved 93.12% accuracy,
480 MB VRAM, or 67,500 tokens/s.

## Other result snapshots

`../results/roberta/summary.csv` is a separate adapted-RoBERTa run. It records accuracy and
elapsed time, but not the VRAM, F1, or throughput values used by the old “champion” paragraph.
Its `fusion_attn` accuracy is 93.12%, so that number must be labeled as an adapted-RoBERTa
result and must not be combined with DistilBERT hardware metrics.

`master_verified_results.json` is a consolidated cross-run index. It reports 331 unique module
names, 262 pass, 7 pass-low, and 62 fail entries. One indexed source path currently points to
the missing `results/unified_adaptive_span/summary.csv`; this index needs regeneration before
being treated as a fully closed audit trail.

## Citation rule

Every thesis table should state the result snapshot, model family, dataset, and source CSV.
Rows with `FAIL`, `TIMEOUT`, `SKIP`, or zero-valued metrics must not be described as successful
benchmarks. A result that is not present in a source CSV is marked `Pending verification`, not
filled in by interpolation or a different run.
