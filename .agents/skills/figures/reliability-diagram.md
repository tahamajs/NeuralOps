# Reliability Diagram

## Purpose

Show calibration: does confidence match accuracy?

## Implementation

```python
def reliability_diagram(confidences, correct, n_bins=15, out_png=None):
    import numpy as np, matplotlib.pyplot as plt
    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_count = []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_count.append(mask.sum())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(bin_conf, bin_acc, width=1.0 / n_bins, alpha=0.7, edgecolor="black")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend()
    if out_png:
        fig.savefig(out_png)
    plt.close(fig)
```

## Output

- `papers/modules/<module>/figures/reliability_diagram.png`
- `papers/modules/<module>/metrics/ece.json`
