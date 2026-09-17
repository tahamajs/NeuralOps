# Failure Signature — svd_projection_attention

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
grep -E "svd_projection_attention|FAIL|ERROR|NaN|OOM" results/*/svd_projection_attention*/summary.csv

# Check for the specific signature
grep "<specific_string>" benchmark_logs/svd_projection_attention*.log
```

## Recovery procedure

1. Read the log in `benchmark_logs/svd_projection_attention_<backbone>.log`
2. Check `metrics/sanity_check.json` for this run
3. Check `metrics/copy_weights_audit.json`
4. If any of the above failed, apply the fix and re-run
5. Do NOT proceed to recovery if sanity check failed

## Output

`papers/modules/svd_projection_attention/logs/failure_analysis.md`
