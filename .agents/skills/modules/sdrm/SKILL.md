# Skill — SDRM

## Purpose
Scaled Dimension Rank Modulation — adaptive rank per layer.

## Rank argument
`total_param_ratio` (not rank directly)

## Known failure
- Adaptive rank silently selects full rank -> no compression
- `max_error` too permissive -> fallback never triggers

## Special handling
- `04-rank-arg-mapping.md` (arg name is different)
- `05-parameter-count-verify.md`
- `10-honest-reporting.md`

## Skip on
Nothing — but verify actual rank chosen per layer
