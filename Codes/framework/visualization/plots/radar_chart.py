import matplotlib.pyplot as plt
import numpy as np


def plot_radar_chart(history, metrics, config, output_dir, opts) -> Any:
    """Plot radar chart."""
    if not metrics:
        return []
    categories = list(metrics.keys())
    values = list(metrics.values())
    # Only numeric values
    numeric_cats = []
    numeric_vals = []
    for c, v in zip(categories, values):
        if isinstance(v, (int, float)):
            numeric_cats.append(c)
            numeric_vals.append(v)
    if len(numeric_cats) < 3:
        return []

    angles = np.linspace(0, 2 * np.pi, len(numeric_cats), endpoint=False).tolist()
    values_norm = numeric_vals + numeric_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=opts.get("figsize", [8, 8]), subplot_kw=dict(polar=True)
    )
    ax.fill(angles, values_norm, alpha=0.25, color="#3B528B")
    ax.plot(angles, values_norm, color="#3B528B", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(numeric_cats)
    ax.set_title("Performance Radar Chart")
    plt.tight_layout()
    path = output_dir / "radar_chart.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
