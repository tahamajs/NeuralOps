# framework/utils/metrics.py

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute and return accuracy."""
    return np.mean(preds == labels)


def compute_matthews_corrcoef(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute and return matthews corrcoef."""
    from sklearn.metrics import matthews_corrcoef

    return matthews_corrcoef(labels, preds)


def compute_pearson(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute and return pearson."""
    return pearsonr(preds, labels)[0]


def compute_spearman(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute and return spearman."""
    return spearmanr(preds, labels)[0]


def compute_f1(preds: np.ndarray, labels: np.ndarray, average: str = "weighted") -> float:
    """Compute and return f1."""
    from sklearn.metrics import f1_score

    return f1_score(labels, preds, average=average, zero_division=0)


def compute_precision_recall(
    preds: np.ndarray, labels: np.ndarray, average: str = "weighted"
) -> dict[str, float]:
    """Compute and return precision recall."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0
    )
    return {"precision": precision, "recall": recall}


def compute_metrics_for_task(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    is_regression: bool = False,
) -> dict[str, float]:
    """Compute and return metrics for task."""
    metrics = {}
    if is_regression or task_name == "stsb":
        pred_flat = np.squeeze(predictions)
        labels_flat = np.squeeze(labels)
        metrics["pearson"] = compute_pearson(pred_flat, labels_flat)
        metrics["spearman"] = compute_spearman(pred_flat, labels_flat)
        metrics["pearson_spearman"] = (metrics["pearson"] + metrics["spearman"]) / 2
    else:
        preds = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
        metrics["accuracy"] = compute_accuracy(preds, labels)
        metrics["f1_macro"] = compute_f1(preds, labels, average="macro")
        metrics["f1_weighted"] = compute_f1(preds, labels, average="weighted")
        if task_name == "cola":
            metrics["matthews_correlation"] = compute_matthews_corrcoef(preds, labels)
        if task_name in ("mrpc", "qqp"):
            metrics["f1"] = compute_f1(preds, labels, average="binary")
    return metrics
