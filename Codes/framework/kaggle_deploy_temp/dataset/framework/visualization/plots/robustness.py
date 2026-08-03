"""Robustness curve – accuracy vs. perturbation magnitude."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_robustness(history, metrics, config, output_dir, opts) -> Any:
    """Plot robustness."""
    rob_path = Path(output_dir).parent / "artifacts" / "robustness.json"
    if not rob_path.exists():
        logger.info("Robustness plot requires robustness.json (perturbation data).")
        return []
    with open(rob_path) as f:
        data = json.load(f)
    perturbations = data.get("perturbations", [])
    accuracies = data.get("accuracies", [])
    baseline_acc = data.get("baseline_accuracy", accuracies[0] if accuracies else 0)
    if len(perturbations) == 0:
        return []

    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 6]))
    ax.plot(
        perturbations,
        accuracies,
        "o-",
        color="#440154",
        markersize=8,
        linewidth=2,
        zorder=5,
    )
    ax.axhline(
        y=baseline_acc,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Baseline ({baseline_acc:.3f})",
    )
    ax.fill_between(perturbations, 0, accuracies, alpha=0.1, color="#440154")
    ax.set_xlabel("Perturbation Magnitude (ε)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Robustness to Perturbation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = str(Path(output_dir) / "robustness_curve.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out]
