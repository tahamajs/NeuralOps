# Skill — CompressedMHA

## Purpose
Joint truncated SVD on stacked Q/K/V projections.

## Rank argument
`rank_qkv` and `rank_out`

## Known failure
- `is_nested_encoder` wrong for DistilBERT (thinks it's nested)
- `copy_weights` raises -> random init
- `head_mask` bound to `attention_mask` on DistilBERT

## Special handling
- `01-copy-weights-audit.md` (essential)
- `02-nested-encoder-check.md` (critical)
- `06-forward-sanity-check.md` (required)

## Skip on
Nothing — but requires correct `is_nested_encoder`
