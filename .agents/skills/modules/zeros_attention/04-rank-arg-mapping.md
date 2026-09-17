# Rank Argument Mapping — zeros_attention

## Purpose

Different modules name the rank argument differently. Using the wrong name
silently defaults and the rank ablation becomes meaningless.

## Map

```python
RANK_ARG_BY_MODULE = {
    "zeros_attention": "<rank_arg_name>",
}
```

Common names:
- `rank`
- `svd_rank`
- `cur_rank`
- `rank_qkv`
- `rank_out`
- `r`

## Verification

Before running, print the constructor signature:

```python
import inspect
sig = inspect.signature(<MODULE_CLASS>.__init__)
print(list(sig.parameters.keys()))
```

If the expected rank arg is not in the signature, do NOT proceed.

## Output

```json
{
  "module": "zeros_attention",
  "rank_arg": "<rank_arg_name>",
  "default_value": 64,
  "min": 1,
  "max": 768
}
```
