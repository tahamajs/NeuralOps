# Per-Layer Reconstruction Error

## Purpose

Show where compression hurts most.

## X-axis

Layer index (0 to N-1)

## Y-axis

Reconstruction error `‖W - UV‖_F / ‖W‖_F`

## Implementation

```python
def plot_per_layer_error(layer_df, out_png):
    fig, ax = plt.subplots(figsize=(8, 4))
    for bb in ["distilbert", "roberta", "gpt2"]:
        sub = layer_df[layer_df.backbone == bb]
        ax.plot(sub["layer_idx"], sub["error"], marker="o", label=bb)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Relative Frobenius error")
    ax.axhline(0.5, ls="--", color="gray", label="Fallback threshold")
    ax.legend()
    fig.savefig(out_png)
    plt.close(fig)
```

## Output

- `papers/modules/<module>/figures/per_layer_error.png`
- Source: `papers/modules/<module>/tables/per_layer_stats.csv`
