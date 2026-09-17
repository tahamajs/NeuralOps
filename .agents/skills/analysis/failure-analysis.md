# Failure Analysis

## Purpose

Systematic analysis of what goes wrong and why.

## Categories

| Category | Detection |
|---|---|
| OOM | "CUDA out of memory" in log |
| NaN | "NaN" in loss log |
| Crash | Python traceback in log |
| Timeout | elapsed > threshold |
| Skip (causal) | Explicit SKIP message |
| Skip (weights) | UniversalAdapter failed |

## Root cause per case

For each failure, record:
- File/line of the failure
- The module's forward pass state at failure
- Prior successful steps
- Hypothesis for why it failed
- Fix attempted

## Output

- `papers/modules/<module>/tables/failure_taxonomy.md`
- `papers/modules/<module>/logs/nan_inf.log`
- `papers/modules/<module>/logs/oom.log`
