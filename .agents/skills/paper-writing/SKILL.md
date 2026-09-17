# Skill — Paper Writing

## Purpose

Assemble a complete, submission-ready paper from experimental artifacts
(metrics, tables, figures) produced by the `experiments/`, `metrics/`, and
`figures/` skills.

## Prerequisites

- All experimental runs completed and saved under `results/`
- `metrics/all_runs.json` present for the target module
- Figures generated in `papers/modules/<module>/figures/`
- Tables generated in `papers/modules/<module>/tables/`

## Output

- `papers/modules/<module>/paper.md` — complete paper draft

## Structure

Every paper has these sections (in order):

1. Abstract — `abstract-template.md`
2. Introduction — `introduction-template.md`
3. Related Work — `related-work-template.md`
4. Method — `method-template.md`
5. Experimental Setup — part of `results-template.md`
6. Results — `results-template.md`
7. Analysis — `analysis-template.md`
8. Limitations — `limitations-template.md`
9. Conclusion — `conclusion-template.md`
10. Reproducibility — `reproducibility-template.md`

## Rules

- **Never invent numbers.** Every number must come from a saved CSV/JSON.
- If a metric is missing, write `N/A (not collected)`.
- Do not claim improvement unless `p_value < 0.05` from
  `metrics/significance_tests.csv`.
- Every figure must be referenced in the text by filename.
- Every table must have a caption.

## Validation

Before declaring done:
- [ ] No `{{PLACEHOLDER}}` remains
- [ ] Every figure path exists
- [ ] Every table path exists
- [ ] Every numeric claim traceable to a source file
