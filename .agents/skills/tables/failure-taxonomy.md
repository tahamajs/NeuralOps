# Failure Taxonomy Table

## Columns

| Module | Backbone | Rank | Error type | Message | Failed at | Elapsed |
|---|---|---|---|---|---|---|

## Error types

- `OOM` — CUDA out of memory
- `NAN` — NaN in loss or activations
- `CRASH` — Python traceback
- `TIMEOUT` — exceeded time budget
- `SKIP_NON_CAUSAL_BACKBONE` — module skipped intentionally
- `SKIP_MISSING_WEIGHTS` — weight extraction failed

## Output

- `papers/modules/<module>/tables/failures.csv`
- `papers/modules/<module>/tables/failure_taxonomy.md`
