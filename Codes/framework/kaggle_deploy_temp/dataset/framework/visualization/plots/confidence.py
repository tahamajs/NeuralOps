"""
Confidence plots – histogram of prediction confidence, calibration curve,
and reliability diagram.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def plot_confidence(history, metrics, config, output_dir, opts) -> Any:
    """Plot confidence."""
    preds_path = Path(output_dir).parent / "artifacts" / "eval_predictions.json"
    if not preds_path.exists():
        logger.info("Confidence plot requested but no eval_predictions.json found.")
        return []

    with open(preds_path) as f:
        data = json.load(f)
    labels = np.array(data.get("labels", []))
    probs = np.array(data.get("probabilities", []))
    if len(labels) == 0 or len(probs) == 0:
        return []

    # Convert to predicted class and confidence
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correct = (predictions == labels).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=opts.get("figsize", [18, 5]))
    fig.suptitle("Confidence Analysis", fontsize=14, fontweight="bold")

    # 1. Confidence histogram (separate correct / incorrect)
    ax = axes[0]
    ax.hist(
        confidences[correct == 1], bins=20, alpha=0.6, label="Correct", color="#21908C"
    )
    ax.hist(
        confidences[correct == 0],
        bins=20,
        alpha=0.6,
        label="Incorrect",
        color="#440154",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Calibration curve (reliability diagram)
    ax = axes[1]
    prob_true, prob_pred = calibration_curve(
        correct, confidences, n_bins=10, strategy="uniform"
    )
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#3B528B")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Correct Predictions")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.grid(True, alpha=0.3)

    # 3. ECE bar
    ax = axes[2]
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(len(bin_boundaries) - 1):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(correct[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(acc_in_bin - conf_in_bin)
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_sizes.append(np.sum(in_bin))
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_sizes.append(0)

    x = np.arange(len(bin_boundaries) - 1)
    width = 0.35
    ax.bar(x - width / 2, bin_accs, width, label="Accuracy", color="#5DC963")
    ax.bar(x + width / 2, bin_confs, width, label="Confidence", color="#FDE725")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}" for i in x], rotation=45
    )
    ax.set_xlabel("Confidence Bin")
    ax.set_ylabel("Value")
    ax.set_title(f"Expected Calibration Error (ECE): {ece:.4f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = str(Path(output_dir) / "confidence_analysis.png")
    fig.savefig(out_path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out_path]
