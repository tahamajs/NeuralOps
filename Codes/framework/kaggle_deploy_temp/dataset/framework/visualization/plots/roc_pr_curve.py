"""
ROC and Precision-Recall curves – industry standard for binary classification.
Produces two subplots: (1) ROC with AUC, (2) Precision-Recall with AP.
"""
from typing import Any

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)


def plot_roc_pr_curve(history, metrics, config, output_dir, opts) -> Any:
    """Plot roc pr curve."""
    preds_path = Path(output_dir).parent / "artifacts" / "eval_predictions.json"
    if not preds_path.exists():
        logger.info("ROC/PR plot requires eval_predictions.json")
        return []
    with open(preds_path) as f:
        data = json.load(f)
    labels = np.array(data.get("labels", []))
    probs = np.array(data.get("probabilities", []))
    if len(labels) == 0 or probs.ndim < 2:
        return []

    n_classes = probs.shape[1]
    class_names = data.get("class_names", [str(i) for i in range(n_classes)])

    fig, axes = plt.subplots(1, 2, figsize=opts.get("figsize", [16, 6]))
    # -- ROC --
    ax = axes[0]
    for i in range(n_classes):
        if n_classes == 2 and i == 0:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                lw=2,
                label=f"{class_names[1]} (AUC={roc_auc:.3f})",
                color="#3B528B",
            )
        elif n_classes > 2:
            y_bin = (labels == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- Precision-Recall --
    ax = axes[1]
    for i in range(n_classes):
        if n_classes == 2 and i == 1:
            precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
            ap = average_precision_score(labels, probs[:, 1])
            ax.plot(
                recall,
                precision,
                lw=2,
                label=f"{class_names[1]} (AP={ap:.3f})",
                color="#21908C",
            )
        elif n_classes > 2:
            y_bin = (labels == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_bin, probs[:, i])
            ap = average_precision_score(y_bin, probs[:, i])
            ax.plot(recall, precision, lw=1.5, label=f"{class_names[i]} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = str(Path(output_dir) / "roc_pr_curves.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out]
