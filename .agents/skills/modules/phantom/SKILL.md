# Skill — Phantom Attention

## Purpose
CUR decomposition + multi-round angular LSH.

## Rank argument
`rank` (CUR) and `bucket_size` (LSH)

## Known failure
- LSH bucket size too large -> dense attention
- LSH bucket size too small -> no context
- Multi-round averaging increases latency 4x

## Special handling
- `08-inference-cost-breakdown.md`
- `09-failure-signature.md`
- `10-honest-reporting.md`

## Skip on
Nothing — but tune `bucket_size` for L
