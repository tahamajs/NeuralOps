"""
Unified Evaluation and Training Metrics Library.
Computes classification, regression, language modeling, representation fidelity,
and hardware efficiency metrics (Perplexity, Accuracy, Latency, Peak VRAM,
Compression Ratio, CKA, Cosine Similarity).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


def compute_accuracy(predictions: Any, labels: Any) -> float:
    """Compute and return standard classification accuracy."""
    return float(accuracy_score(labels, predictions))


def compute_f1(predictions: Any, labels: Any, average: str = "weighted") -> float:
    """Compute and return F1 score."""
    return float(f1_score(labels, predictions, average=average, zero_division=0))


def compute_matthews_corrcoef(predictions: Any, labels: Any) -> float:
    """Compute and return Matthews correlation coefficient."""
    return float(matthews_corrcoef(labels, predictions))


def compute_pearson_spearman(predictions: Any, labels: Any) -> Dict[str, float]:
    """Compute and return Pearson, Spearman, and average correlation."""
    pred_flat = np.squeeze(predictions)
    labels_flat = np.squeeze(labels)
    p = float(pearsonr(pred_flat, labels_flat)[0])
    s = float(spearmanr(pred_flat, labels_flat)[0])
    return {"pearson": p, "spearman": s, "pearson_spearman": (p + s) / 2.0}


def compute_precision_recall(predictions: Any, labels: Any, average: str = "weighted") -> Dict[str, float]:
    """Compute precision and recall."""
    p, r, _, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return {"precision": float(p), "recall": float(r)}


def compute_regression_metrics(predictions: Any, labels: Any) -> Dict[str, float]:
    """Compute MSE, MAE, and RMSE for regression outputs."""
    mse = float(mean_squared_error(labels, predictions))
    mae = float(mean_absolute_error(labels, predictions))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from negative log-likelihood loss with overflow protection."""
    if not math.isfinite(loss):
        return float("inf")
    return float(math.exp(min(loss, 100.0)))


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> float:
    """Compute mean cosine similarity between two representation tensors."""
    if a.shape != b.shape:
        min_dim = min(a.shape[-1], b.shape[-1])
        a = a[..., :min_dim]
        b = b[..., :min_dim]
    sim = F.cosine_similarity(a.float(), b.float(), dim=dim)
    return float(sim.mean().item())


def compute_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Linear Centered Kernel Alignment (CKA) between representations x and y.
    """
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    if y.dim() > 2:
        y = y.reshape(-1, y.shape[-1])

    x = x.float()
    y = y.float()

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    dot_xy = torch.norm(torch.matmul(x.t(), y)) ** 2
    dot_xx = torch.norm(torch.matmul(x.t(), x)) ** 2
    dot_yy = torch.norm(torch.matmul(y.t(), y)) ** 2

    denom = torch.sqrt(dot_xx * dot_yy)
    if denom.item() == 0:
        return 0.0
    return float((dot_xy / denom).item())


def compute_compression_ratio(original_params: int, compressed_params: int) -> float:
    """Compute factor of parameter reduction."""
    if compressed_params <= 0:
        return 1.0
    return float(original_params / compressed_params)


def get_task_metrics(task_name: str) -> Any:
    """Return a compute_metrics function for a GLUE or benchmark task."""
    task_name = task_name.lower()
    if task_name == "stsb":

        def fn(p: Any) -> Dict[str, float]:
            preds = np.squeeze(p.predictions)
            labels = p.label_ids
            return compute_pearson_spearman(preds, labels)

    elif task_name == "cola":

        def fn(p: Any) -> Dict[str, float]:
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            return {
                "matthews_correlation": float(matthews_corrcoef(labels, preds)),
                "accuracy": float(accuracy_score(labels, preds)),
            }

    elif task_name in ("mrpc", "qqp"):

        def fn(p: Any) -> Dict[str, float]:
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            acc = float(accuracy_score(labels, preds))
            f1 = float(f1_score(labels, preds))
            return {"accuracy": acc, "f1": f1, "accuracy_f1": (acc + f1) / 2.0}

    else:

        def fn(p: Any) -> Dict[str, float]:
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
            }

    return fn


def aggregate_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of metric dictionaries into mean, std, and raw values."""
    if not results_list:
        return {}
    aggregated: Dict[str, Any] = {}
    for key in results_list[0].keys():
        values = [r[key] for r in results_list if key in r and isinstance(r[key], (int, float))]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_values"] = values
    return aggregated


__all__ = [
    "compute_accuracy",
    "compute_f1",
    "compute_matthews_corrcoef",
    "compute_pearson_spearman",
    "compute_precision_recall",
    "compute_regression_metrics",
    "compute_perplexity",
    "compute_cosine_similarity",
    "compute_linear_cka",
    "compute_compression_ratio",
    "get_task_metrics",
    "aggregate_results",
]
