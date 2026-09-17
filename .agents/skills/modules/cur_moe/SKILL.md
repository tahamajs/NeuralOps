# Skill — CUR-MoE

## Purpose
CUR matrix decomposition for MoE expert layers.

## Rank argument
`rank`

## Known failure
- Router collapse to single expert
- CUR fails on rank-deficient matrices

## Special handling
- `01-copy-weights-audit.md`
- `06-forward-sanity-check.md`
- `09-failure-signature.md` (router collapse)

## Skip on
Models without MoE structure (falls back to FFN with bad ratio)
