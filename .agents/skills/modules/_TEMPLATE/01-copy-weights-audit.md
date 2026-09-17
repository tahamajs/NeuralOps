# Copy Weights Audit — <MODULE_NAME>

## Purpose

Verify that `copy_weights` actually transfers pretrained weights, not random
initialization. This is the #1 cause of the 50.92% collapse.

## Method

```python
def audit_copy_weights(module, original_attn):
    """Return a dict: {proj: {'shape_ok': bool, 'norm_ratio': float}}."""
    import torch
    report = {}
    for name, target in [("q", module.q_proj), ("k", module.k_proj),
                          ("v", module.v_proj), ("o", module.out_proj)]:
        w_target = target.weight.data
        w_norm = w_target.norm().item()
        # Random init has norm ~ sqrt(fan_in) * 0.02
        random_expected = (w_target.numel() ** 0.5) * 0.02
        ratio = w_norm / random_expected
        report[name] = {
            "norm": w_norm,
            "random_expected": random_expected,
            "ratio": ratio,
            "looks_copied": ratio > 5.0,  # copied weights are much larger
        }
    return report
```

## Verification criteria

| Projection | Expected ratio |
|---|---|
| q, k, v, o | > 10 (copied), < 3 (random) |

If any projection has ratio < 5, `copy_weights` did NOT transfer.

## Root causes to check

1. **Extraction path wrong**: e.g., looking for `q_proj` on a model that has `q_lin`.
2. **Exception swallowed**: `try/except` around the copy hides failure.
3. **Wrong orientation**: `W` vs `W.t()`.
4. **Nested encoder**: copying from `attention.self.query` while the model has `attention.self.q_lin`.
5. **Shape mismatch**: skipped silently.

## Output

Save to `papers/modules/<module>/metrics/copy_weights_audit.json`:

```json
{
  "distilbert": {
    "layer.0.q_proj": {"norm": 12.4, "ratio": 45.2, "looks_copied": true},
    "layer.0.k_proj": {"norm": 11.8, "ratio": 42.1, "looks_copied": true}
  }
}
```
