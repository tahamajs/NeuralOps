# Failure Signature — a3

## Purpose

Document the specific failure pattern of this module. Every agent working with
this module must know it.

## Known failures

### Failure 1: <name>

**Symptom:** <what happens>

**Root cause:** <why>

**Fix:** <how>

**Files affected:** <list>

### Failure 2: <name>

...

## Diagnostic commands

```bash
# Check if the module is even running correctly
grep -E "a3|FAIL|ERROR|NaN|OOM" results/*/a3*/summary.csv

# Check for the specific signature
grep "<specific_string>" benchmark_logs/a3*.log
```

## Recovery procedure

1. Read the log in `benchmark_logs/a3_<backbone>.log`
2. Check `metrics/sanity_check.json` for this run
3. Check `metrics/copy_weights_audit.json`
4. If any of the above failed, apply the fix and re-run
5. Do NOT proceed to recovery if sanity check failed

## Output

`papers/modules/a3/logs/failure_analysis.md`
