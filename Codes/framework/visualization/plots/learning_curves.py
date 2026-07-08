"""
Learning curves – shows training/validation score vs. training set size.
Diagnoses bias/variance and whether more data would help.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_learning_curve(history, metrics, config, output_dir, opts):
    # Requires training data splits captured at multiple sizes.
    # If not available, falls back to using history steps as proxy.
    lc_path = Path(output_dir).parent / "artifacts" / "learning_curve_data.json"
    if lc_path.exists():
        with open(lc_path) as f:
            data = json.load(f)
        train_sizes = data.get("train_sizes", [])
        train_mean = data.get("train_scores_mean", [])
        train_std = data.get("train_scores_std", [])
        val_mean = data.get("val_scores_mean", [])
        val_std = data.get("val_scores_std", [])
    else:
        # Fallback: use steps as a proxy (not true learning curve but useful)
        steps = [h.get("step", i) for i, h in enumerate(history) if "loss" in h]
        train = [h["loss"] for h in history if "loss" in h]
        [h.get("step", i) for i, h in enumerate(history) if "eval_loss" in h]
        eval_l = [h["eval_loss"] for h in history if "eval_loss" in h]
        if not steps:
            return []
        train_sizes = steps
        train_mean = train
        train_std = [0] * len(train)
        val_mean = eval_l
        val_std = [0] * len(eval_l)
        logger.info("No learning_curve_data.json found; using step‑based fallback.")

    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 6]))
    if train_sizes:
        ax.plot(
            train_sizes,
            train_mean,
            "o-",
            color="#3B528B",
            label="Training score",
            linewidth=2,
        )
        ax.fill_between(
            train_sizes,
            np.array(train_mean) - np.array(train_std),
            np.array(train_mean) + np.array(train_std),
            alpha=0.15,
            color="#3B528B",
        )
    if val_mean:
        x_val = (
            train_sizes[: len(val_mean)]
            if len(val_mean) <= len(train_sizes)
            else range(len(val_mean))
        )
        ax.plot(
            x_val,
            val_mean,
            "o-",
            color="#21908C",
            label="Validation score",
            linewidth=2,
        )
        ax.fill_between(
            x_val,
            np.array(val_mean) - np.array(val_std),
            np.array(val_mean) + np.array(val_std),
            alpha=0.15,
            color="#21908C",
        )
    ax.set_xlabel("Training examples / Steps")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curves (Bias-Variance Diagnosis)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = str(Path(output_dir) / "learning_curves.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out]
