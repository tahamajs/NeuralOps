from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


def plot_combined_training_curves(history, metrics, config, output_dir, opts) -> Any:
    """Plot combined training curves."""
    steps = [h.get("step", i) for i, h in enumerate(history) if "loss" in h]
    train = [h["loss"] for h in history if "loss" in h]
    eval_steps = [h.get("step", i) for i, h in enumerate(history) if "eval_loss" in h]
    eval_l = [h["eval_loss"] for h in history if "eval_loss" in h]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    smooth = opts.get("loss_smooth", 0.6)
    if smooth > 0 and len(train) > 5:
        window = max(1, int(len(train) * smooth / 10))
        train = uniform_filter1d(np.array(train), size=window)
    axes[0].plot(steps, train, color="#3B528B")
    if eval_steps:
        axes[0].plot(eval_steps, eval_l, "o-", color="#21908C", markersize=4)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend(["Train", "Eval"])
    axes[0].grid(True, alpha=0.3)

    acc_steps = [
        h.get("step", i) for i, h in enumerate(history) if "eval_accuracy" in h
    ]
    acc = [h["eval_accuracy"] for h in history if "eval_accuracy" in h]
    if acc_steps:
        axes[1].plot(acc_steps, acc, "o-", color="#5DC963", markersize=4)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No accuracy data", ha="center", va="center")
    plt.tight_layout()
    path = Path(output_dir) / "combined_curves.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
