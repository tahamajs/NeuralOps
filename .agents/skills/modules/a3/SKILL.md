# Skill — A3

## Purpose
Activation-aware SVD compression with Fisher weighting.

## Rank argument
`rank`

## Known failure
- Without calibration data, weighting is a no-op
- `max_error` too tight -> rank escalates to full

## Special handling
- Calibration must be provided (else skip or use default corpus)
- `10-honest-reporting.md` (params decrease is real)

## Skip on
Nothing — but check calibration source
