# Parameter Count Verification — jamba2

## Purpose

Confirm that the module actually reduces parameters. Many modules claim
"compression" but increase parameter count.

## Method

```python
def verify_parameter_reduction(model_before, model_after):
    p_before = sum(p.numel() for p in model_before.parameters())
    p_after = sum(p.numel() for p in model_after.parameters())
    change_pct = 100.0 * (p_after - p_before) / p_before
    return {
        "before": p_before,
        "after": p_after,
        "change_pct": change_pct,
        "is_compression": change_pct < 0,
    }
```

## Expected for jamba2

<what the module claims>

## Actual (fill in from runs)

| Backbone | Before | After | Change % |
|---|---|---|---|
| DistilBERT | 66955010 | ? | ? |
| RoBERTa | 124647170 | ? | ? |
| GPT-2 | 124441344 | ? | ? |

## Honest reporting

If `change_pct > 0`:
- Rename in the paper: "re-architecture", not "compression"
- State the increase explicitly in the abstract
- Do NOT claim Pareto improvement

## Output

`papers/modules/jamba2/metrics/parameter_count.json`
