"""Activation distribution – histogram of layer activations."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_activation_distribution(history, metrics, config, output_dir, opts):
    act_path = Path(output_dir).parent / "artifacts" / "activations.json"
    if not act_path.exists():
        logger.info("Activation distribution requires activations.json.")
        return []
    with open(act_path) as f:
        data = json.load(f)
    layers = data if isinstance(data, dict) else {"all": data}
    n = len(layers)
    if n == 0:
        return []

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    for idx, (name, acts) in enumerate(layers.items()):
        ax = axes[idx // cols][idx % cols]
        vals = np.array(acts).flatten()
        vals = vals[~np.isnan(vals)] if len(vals) > 0 else vals
        ax.hist(vals, bins=50, color="#21908C", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"{name} (μ={vals.mean():.2f}, σ={vals.std():.2f})", fontsize=9)
        ax.set_xlabel("Activation")
        ax.set_ylabel("Count")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle("Activation Distributions per Layer", fontsize=12)
    plt.tight_layout()
    out = str(Path(output_dir) / "activation_distribution.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out]
