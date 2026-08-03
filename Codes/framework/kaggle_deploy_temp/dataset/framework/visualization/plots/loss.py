import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


def plot_loss_curve(history, metrics, config, output_dir, opts) -> Any:
    """Plot loss curve."""
    steps = [h.get("step", i) for i, h in enumerate(history) if "loss" in h]
    train = [h["loss"] for h in history if "loss" in h]
    eval_steps = [h.get("step", i) for i, h in enumerate(history) if "eval_loss" in h]
    eval_l = [h["eval_loss"] for h in history if "eval_loss" in h]

    if not steps:
        logger.info("No loss data found – skipping loss curve")
        return []

    smooth = opts.get("loss_smooth", 0.6)
    if smooth > 0 and len(train) > 5:
        window = max(1, int(len(train) * smooth / 10))
        train = uniform_filter1d(np.array(train), size=window)

    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 6]))
    ax.plot(steps, train, label="Train Loss", color="#3B528B")
    if eval_steps:
        ax.plot(
            eval_steps, eval_l, "o-", label="Eval Loss", color="#21908C", markersize=4
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "loss_curve.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
