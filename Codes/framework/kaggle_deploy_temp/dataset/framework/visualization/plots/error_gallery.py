"""
Error gallery – shows the top‑K worst predictions with their inputs, true labels,
and predicted probabilities.
"""
from typing import Any

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_error_gallery(history, metrics, config, output_dir, opts) -> Any:
    """Plot error gallery."""
    preds_path = Path(output_dir).parent / "artifacts" / "eval_predictions.json"
    if not preds_path.exists():
        logger.info("Error gallery requested but no eval_predictions.json found.")
        return []

    with open(preds_path) as f:
        data = json.load(f)
    texts = data.get("texts", [])  # raw input texts
    labels = np.array(data.get("labels", []))
    predictions = np.array(data.get("predictions", []))  # predicted class indices
    probs = np.array(data.get("probabilities", []))  # full probability vectors
    class_names = data.get("class_names", None)

    if len(labels) == 0 or len(predictions) == 0:
        return []

    # Find the worst‑error samples (highest confidence on wrong answer)
    errors_mask = predictions != labels
    if not np.any(errors_mask):
        logger.info("No errors found – skipping error gallery")
        return []

    wrong_probs = probs[errors_mask]
    wrong_preds = predictions[errors_mask]
    wrong_labels = labels[errors_mask]
    wrong_texts = [texts[i] for i in range(len(texts)) if errors_mask[i]]

    # Confidence of incorrect prediction = max probability given to the wrong class
    wrong_confidences = wrong_probs[np.arange(len(wrong_preds)), wrong_preds]
    top_k_idx = np.argsort(wrong_confidences)[-opts.get("error_gallery_k", 8) :][::-1]

    k = min(opts.get("error_gallery_k", 5), len(top_k_idx))
    fig, axes = plt.subplots(k, 1, figsize=(10, 3 * k), squeeze=False)
    axes = axes.flatten()

    for ax_idx, wrong_idx in enumerate(top_k_idx[:k]):
        ax = axes[ax_idx]
        text = wrong_texts[wrong_idx] if wrong_texts else f"Sample {wrong_idx}"
        true_label = (
            class_names[wrong_labels[wrong_idx]]
            if class_names
            else str(wrong_labels[wrong_idx])
        )
        pred_label = (
            class_names[wrong_preds[wrong_idx]]
            if class_names
            else str(wrong_preds[wrong_idx])
        )
        conf = wrong_confidences[wrong_idx]

        # Truncate long texts
        display_text = text[:200] + "..." if len(text) > 200 else text
        ax.text(0.05, 0.7, display_text, fontsize=9, wrap=True)
        ax.text(
            0.05,
            0.4,
            f"True: {true_label}  |  Predicted: {pred_label}  |  Confidence: {conf:.3f}",
            fontsize=10,
            fontweight="bold",
            color="red",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Top {k} High‑Confidence Errors", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = str(Path(output_dir) / "error_gallery.png")
    fig.savefig(out_path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out_path]
