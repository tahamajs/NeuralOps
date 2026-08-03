"""Precision-Recall per class bar chart."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def plot_precision_recall_by_class(history, metrics, config, output_dir, opts) -> Any:
    """Plot precision recall by class."""
    preds_path = Path(output_dir).parent / "artifacts" / "eval_predictions.json"
    if not preds_path.exists():
        return []
    with open(preds_path) as f:
        data = json.load(f)
    labels = np.array(data.get("labels", []))
    predictions = np.array(data.get("predictions", []))
    class_names = data.get("class_names", None)
    if len(labels) == 0:
        return []
    unique = np.unique(np.concatenate([labels, predictions]))
    if class_names is None:
        class_names = [str(int(u)) for u in unique]

    prec = precision_score(
        labels, predictions, average=None, labels=unique, zero_division=0
    )
    rec = recall_score(
        labels, predictions, average=None, labels=unique, zero_division=0
    )
    f1 = f1_score(labels, predictions, average=None, labels=unique, zero_division=0)

    x = np.arange(len(unique))
    width = 0.25
    fig, ax = plt.subplots(figsize=opts.get("figsize", [12, 5]))
    ax.bar(
        x - width, prec, width, label="Precision", color="#3B528B", edgecolor="white"
    )
    ax.bar(x, rec, width, label="Recall", color="#21908C", edgecolor="white")
    ax.bar(x + width, f1, width, label="F1", color="#5DC963", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall, F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = str(Path(output_dir) / "precision_recall_by_class.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out]
