# Causal vs Bidirectional — aasvd

## Purpose

Some modules implement causal recurrence that is mathematically incompatible
with bidirectional encoders. Applying them to BERT produces NaN or 0.00%.

## Detection

```python
CAUSAL_ONLY_MODULES = {
    "gated_delta_net", "gated_deltanet_block",
    "hgrn2_gated_deltanet", "hgrn2_layer",
    "jamba2", "jamba",
    "decay_gated_linear_attention",
    "svd_mamba", "mamba", "mamba2",
}

def is_compatible(model_family, module_name):
    if module_name in CAUSAL_ONLY_MODULES and model_family in ("bert", "roberta"):
        return False, "SKIP_NON_CAUSAL_BACKBONE"
    return True, None
```

## Rule

If `aasvd` ∈ `CAUSAL_ONLY_MODULES` and model_family ∈ {bert, roberta}:
- Mark status as `SKIP_NON_CAUSAL_BACKBONE`
- Do NOT run recovery
- Log in `papers/modules/aasvd/tables/skip_reasons.csv`

## Output

```csv
module,backbone,reason
aasvd,distilbert,SKIP_NON_CAUSAL_BACKBONE
aasvd,roberta,SKIP_NON_CAUSAL_BACKBONE
```
