import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_cm

logger = logging.getLogger(__name__)


def plot_confusion_matrix(history, metrics, config, output_dir, opts) -> Any:
    """Plot confusion matrix."""
    preds_path = Path(output_dir).parent / "artifacts" / "eval_predictions.json"
    if not preds_path.exists():
        logger.info("Confusion matrix requested but no eval_predictions.json found.")
        return []
    with open(preds_path) as f:
        data = json.load(f)
    labels = data.get("labels", [])
    predictions = data.get("predictions", [])
    if not labels or not predictions:
        return []

    cm = sklearn_cm(labels, predictions)
    class_names = opts.get("class_names", None)
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 8]))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
