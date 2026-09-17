# Skill — Metrics

## Purpose

Compute statistical and diagnostic metrics beyond accuracy and F1.

## Sub-skills

- `statistical-significance.md` — t-tests, McNemar, bootstrap CI
- `spectral-analysis.md` — SVD spectra, effective rank, condition number
- `attention-analysis.md` — entropy, sink mass, head redundancy
- `output-analysis.md` — calibration, agreement, KL divergence
- `training-dynamics.md` — loss curves, gradient norms, convergence
- `resource-accounting.md` — GPU-hours, CO2, FLOPs

## Rules

- Every metric must be traceable to a script
- Never report a single number without uncertainty
- Always report n (number of seeds, samples, or runs)
