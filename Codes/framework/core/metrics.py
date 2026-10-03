"""
core/metrics.py — Dataset-specific metric computation for NeurIPS/ICLR publication.
Supports: Accuracy, F1, MCC (CoLA), Pearson/Spearman (STS-B), Perplexity (LM).
"""
from __future__ import annotations

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)

GLUE_METRIC_MAP = {
    "cola": "mcc",
    "stsb": "pearson_spearman",
    "sts-b": "pearson_spearman",
    "mrpc": "f1_accuracy",
    "qqp": "f1_accuracy",
    "sst2": "accuracy",
    "sst-2": "accuracy",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "wnli": "accuracy",
    "wikitext2": "perplexity",
    "wikitext103": "perplexity",
    "c4": "perplexity",
}


def compute_metrics(task_name: str, predictions: list[Any], labels: list[Any], mean_loss: float = 0.0) -> dict[str, Any]:
    """Compute dataset-specific metrics using sklearn/scipy."""
    return compute_dataset_metrics(task_name, labels, predictions, mean_loss)


def compute_dataset_metrics(dataset: str, all_labels: list[Any], all_preds: list[Any], mean_loss: float = 0.0) -> dict[str, Any]:
    """Compute the correct metrics for each GLUE task."""
    metrics = {"accuracy_pct": 0.0, "f1_score": 0.0, "mcc": 0.0, "pearson": 0.0, "spearman": 0.0, "perplexity": 0.0}

    if not all_labels or not all_preds:
        return metrics

    ds = dataset.lower().replace("-", "")
    metric_type = GLUE_METRIC_MAP.get(ds, GLUE_METRIC_MAP.get(dataset.lower(), "accuracy"))

    try:
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
        from scipy.stats import pearsonr, spearmanr
    except ImportError as e:
        logger.warning("sklearn/scipy not available: %s", e)
        return metrics

    try:
        if metric_type == "mcc":
            metrics["mcc"] = round(float(matthews_corrcoef(all_labels, all_preds)), 4)
            metrics["accuracy_pct"] = round(float(accuracy_score(all_labels, all_preds)) * 100.0, 2)

        elif metric_type == "pearson_spearman":
            p, _ = pearsonr(all_labels, all_preds)
            s, _ = spearmanr(all_labels, all_preds)
            metrics["pearson"] = round(float(p), 4)
            metrics["spearman"] = round(float(s), 4)

        elif metric_type == "f1_accuracy":
            labels_set = set(all_labels)
            avg = "binary" if len(labels_set) <= 2 else "macro"
            metrics["accuracy_pct"] = round(float(accuracy_score(all_labels, all_preds)) * 100.0, 2)
            metrics["f1_score"] = round(float(f1_score(all_labels, all_preds, average=avg, zero_division=0)), 4)

        elif metric_type == "perplexity":
            if mean_loss > 0:
                metrics["perplexity"] = round(float(math.exp(min(mean_loss, 20))), 4)

        else:  # default: accuracy
            metrics["accuracy_pct"] = round(float(accuracy_score(all_labels, all_preds)) * 100.0, 2)
            labels_set = set(all_labels)
            avg = "binary" if len(labels_set) <= 2 else "macro"
            try:
                metrics["f1_score"] = round(float(f1_score(all_labels, all_preds, average=avg, zero_division=0)), 4)
            except Exception:
                pass

    except Exception as e:
        logger.warning("Metric computation failed for dataset '%s': %s", dataset, e)

    return metrics


def compute_fidelity(baseline_logits_list: list[Any], compressed_logits_list: list[Any]) -> float:
    """Compute cosine-similarity fidelity between baseline and compressed logits."""
    if not baseline_logits_list or not compressed_logits_list:
        return 0.0
    try:
        import torch
        import torch.nn.functional as F
        b_cat = torch.cat(baseline_logits_list, dim=0).view(-1)
        c_cat = torch.cat(compressed_logits_list, dim=0).view(-1)
        sim = F.cosine_similarity(b_cat.unsqueeze(0), c_cat.unsqueeze(0), dim=1)
        return round(float(sim.item()), 4)
    except Exception as e:
        logger.warning("Fidelity computation failed: %s", e)
        return 0.0
