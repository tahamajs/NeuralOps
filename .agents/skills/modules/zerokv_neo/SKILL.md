# Skill — ZeroKV-Neo

## Purpose
L2-norm KV cache eviction with attention sink and heavy-hitter preservation.

## Rank argument
`keep_ratio` (not `rank`)

## Known failure
- `_is_attention_proj` matches `out_proj` -> destroys output projection
- `new_layer` tautology -> no actual compression
- `output_scale` unused parameter

## Special handling
- `01-copy-weights-audit.md` (verify q/k/v copied, o NOT targeted)
- `09-failure-signature.md` (see `_is_attention_proj` bug)
- `10-honest-reporting.md` (params increase due to LoRA)

## Skip on
Nothing — but verify `keep_ratio >= 0.5` in config
