# Skill — AA-SVD

## Purpose
Activation-aware SVD with sensitivity calibration.

## Rank argument
`rank`

## Known failure
- Missing activation hooks lead to unweighted SVD
- `output_scale` scalar adds 1 extra parameter (32769 vs 32768)

## Special handling
- Calibration activation cache must be present
- `05-parameter-count-verify.md`
