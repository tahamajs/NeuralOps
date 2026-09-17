# Skill — Figures

## Purpose

Generate publication-quality figures at 300 DPI.

## Style guide

```python
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
```

## Sub-skills

- `rank-vs-accuracy.md`
- `pareto-frontier.md`
- `training-curves.md`
- `per-layer-error.md`
- `attention-maps.md`
- `reliability-diagram.md`

## Rules

- Always 300 DPI
- Always save both PNG and PDF
- Every axis must have a label with units
- Every line must have a legend entry
- No pie charts
- No 3D plots
