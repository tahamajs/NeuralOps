# Forward Sanity Check — compressed_attention

## Purpose

Before recovery fine-tuning, confirm the injected module produces
non-constant logits.

## When to run

After injection, before recovery.

## Method

```python
def forward_sanity_check(model, sample_ids):
    model.eval()
    with torch.no_grad():
        logits = model(sample_ids).logits  # shape (4, num_classes)
    std = logits.std(dim=0).mean().item()
    unique_preds = logits.argmax(-1).unique().numel()
    return {
        "std": std,
        "unique_preds": unique_preds,
        "passed": std > 1e-3 and unique_preds > 1,
    }
```

## Interpretation

| std | unique_preds | Meaning |
|---|---|---|
| < 1e-3 | 1 | Total collapse — abort |
| < 1e-3 | > 1 | Suspicious — investigate |
| > 1e-3 | 1 | Wrong but not collapsed |
| > 1e-3 | > 1 | Healthy |

## Action on failure

1. Log to `papers/modules/compressed_attention/logs/sanity_fail.log`
2. Do NOT proceed to recovery
3. Mark run as `FAIL_SANITY`
4. Report in `tables/failures.csv`

## Output

`papers/modules/compressed_attention/metrics/sanity_check.json`
