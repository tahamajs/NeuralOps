# Rank vs Accuracy Figure

## Purpose

Show how accuracy changes as rank increases, for each backbone.

## X-axis

Rank (log scale, base 2): [16, 32, 64, 128, 192, 256, 384]

## Y-axis

Accuracy (%) with error bars (std over seeds).

## Horizontal lines

- Majority baseline (e.g., 27.4% on AG News)
- PASS threshold (e.g., 31.63%)

## Implementation

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_rank_vs_accuracy(df, out_png):
    fig, ax = plt.subplots(figsize=(6, 4))
    for bb, color in [("distilbert", "C0"), ("roberta", "C1"), ("gpt2", "C2")]:
        sub = df[df.backbone == bb].sort_values("rank")
        ax.errorbar(sub["rank"], sub["accuracy_mean"],
                    yerr=sub["accuracy_std"], marker="o",
                    capsize=3, label=bb, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Compression Rank")
    ax.set_ylabel("AG News Accuracy (%)")
    ax.axhline(31.63, ls="--", color="gray", label="PASS threshold")
    ax.axhline(27.4, ls=":", color="red", label="Majority baseline")
    ax.legend()
    fig.savefig(out_png)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)
```

## Output

- `papers/modules/<module>/figures/rank_vs_accuracy.png`
- `papers/modules/<module>/figures/rank_vs_accuracy.pdf`
