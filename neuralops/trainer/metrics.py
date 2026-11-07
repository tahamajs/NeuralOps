#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Training and Evaluation Metrics

Computes statistical evaluation metrics, correlation coefficients, accuracy, and classification scores.
"""

from typing import Any

import numpy as np

try:
    import scipy.stats as _sp_stats
    pearsonr = _sp_stats.pearsonr
    spearmanr = _sp_stats.spearmanr
except ImportError:
    def pearsonr(x: Any, y: Any) -> tuple[float, float]:
        x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mx, my = float(np.mean(x_arr)), float(np.mean(y_arr))
        vx, vy = x_arr - mx, y_arr - my
        denom = float(np.sqrt(np.sum(vx**2) * np.sum(vy**2)))
        r = float(np.sum(vx * vy) / max(denom, 1e-12))
        return (r, 0.0)

    def spearmanr(x: Any, y: Any) -> tuple[float, float]:
        x_arr, y_arr = np.asarray(x), np.asarray(y)
        rx = np.argsort(np.argsort(x_arr)).astype(float)
        ry = np.argsort(np.argsort(y_arr)).astype(float)
        return pearsonr(rx, ry)

# Pure-numpy replacements for sklearn.metrics — avoids OpenMP/TLS conflicts on
# CentOS 6 (kernel 2.6.32) when torch and sklearn are loaded in the same process.
# sklearn is NOT imported here; all implementations are equivalent for our usage.


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, label: Any):
    """Return (TP, FP, FN) for a single label."""
    tp = int(np.sum((y_pred == label) & (y_true == label)))
    fp = int(np.sum((y_pred == label) & (y_true != label)))
    fn = int(np.sum((y_pred != label) & (y_true == label)))
    return tp, fp, fn


def accuracy_score(y_true, y_pred) -> float:  # type: ignore[misc]
    """Fraction of correctly classified samples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def mean_squared_error(y_true, y_pred) -> float:  # type: ignore[misc]
    """Mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred) -> float:  # type: ignore[misc]
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def f1_score(y_true, y_pred, average: str = "weighted", zero_division: int = 0) -> Any:  # type: ignore[misc]
    """F1 score (macro / weighted / micro / None)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s, weights = [], []
    for lbl in labels:
        tp, fp, fn = _confusion(y_true, y_pred, lbl)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else float(zero_division)
        f1s.append(f1)
        weights.append(int(np.sum(y_true == lbl)))
    f1s_arr = np.array(f1s, dtype=float)
    w_arr = np.array(weights, dtype=float)
    if average == "macro":
        return float(np.mean(f1s_arr))
    if average == "weighted":
        return float(np.average(f1s_arr, weights=w_arr)) if w_arr.sum() > 0 else float(zero_division)
    if average == "micro":
        tps = sum(_confusion(y_true, y_pred, lbl)[0] for lbl in labels)
        fps = sum(_confusion(y_true, y_pred, lbl)[1] for lbl in labels)
        fns = sum(_confusion(y_true, y_pred, lbl)[2] for lbl in labels)
        d = 2 * tps + fps + fns
        return float(2 * tps / d) if d > 0 else float(zero_division)
    return f1s_arr.tolist()


def matthews_corrcoef(y_true, y_pred) -> float:  # type: ignore[misc]
    """Matthews correlation coefficient (binary classification)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def precision_recall_fscore_support(  # type: ignore[misc]
    y_true, y_pred, average: str = "weighted", zero_division: int = 0
) -> tuple:
    """Precision, recall, F-score, and support."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(y_true)
    ps, rs, f1s, supports = [], [], [], []
    for lbl in labels:
        tp, fp, fn = _confusion(y_true, y_pred, lbl)
        p = tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)
        r = tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)
        d = 2 * tp + fp + fn
        f = (2 * tp / d) if d > 0 else float(zero_division)
        ps.append(p)
        rs.append(r)
        f1s.append(f)
        supports.append(tp + fn)
    ps_arr = np.array(ps, dtype=float)
    rs_arr = np.array(rs, dtype=float)
    f1s_arr = np.array(f1s, dtype=float)
    sup_arr = np.array(supports, dtype=float)
    if average == "weighted":
        w = sup_arr
        return (
            float(np.average(ps_arr, weights=w)) if w.sum() > 0 else float(zero_division),
            float(np.average(rs_arr, weights=w)) if w.sum() > 0 else float(zero_division),
            float(np.average(f1s_arr, weights=w)) if w.sum() > 0 else float(zero_division),
            None,
        )
    if average == "macro":
        return float(np.mean(ps_arr)), float(np.mean(rs_arr)), float(np.mean(f1s_arr)), None
    return ps_arr.tolist(), rs_arr.tolist(), f1s_arr.tolist(), sup_arr.tolist()


def precision_score(y_true, y_pred, average: str = "weighted", zero_division: int = 0) -> float:  # type: ignore[misc]
    """Weighted/macro precision score."""
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=zero_division)
    return float(p)


def recall_score(y_true, y_pred, average: str = "weighted", zero_division: int = 0) -> float:  # type: ignore[misc]
    """Weighted/macro recall score."""
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=zero_division)
    return float(r)


def compute_accuracy(predictions, labels) -> Any:
    """Compute and return accuracy."""
    return accuracy_score(labels, predictions)


def compute_f1(predictions, labels, average="weighted") -> Any:
    """Compute and return f1."""
    return f1_score(labels, predictions, average=average, zero_division=0)


def compute_matthews_corrcoef(predictions, labels) -> Any:
    """Compute and return matthews corrcoef."""
    return matthews_corrcoef(labels, predictions)


def compute_pearson_spearman(predictions, labels) -> Any:
    """Compute and return pearson spearman."""
    pred_flat = np.squeeze(predictions)
    labels_flat = np.squeeze(labels)
    p = pearsonr(pred_flat, labels_flat)[0]
    s = spearmanr(pred_flat, labels_flat)[0]
    return {"pearson": p, "spearman": s, "pearson_spearman": (p + s) / 2}


def compute_precision_recall(predictions, labels, average="weighted") -> Any:
    """Compute and return precision recall."""
    p, r, _, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return {"precision": p, "recall": r}


def compute_regression_metrics(predictions, labels) -> Any:
    """Compute and return regression metrics."""
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def get_task_metrics(task_name: str) -> Any:
    """Return a compute_metrics function for a GLUE task."""
    task_name = task_name.lower()
    if task_name == "stsb":

        def fn(p) -> Any:
            """Fn."""
            preds = np.squeeze(p.predictions)
            labels = p.label_ids
            return compute_pearson_spearman(preds, labels)

    elif task_name == "cola":

        def fn(p) -> Any:
            """Fn."""
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            return {
                "matthews_correlation": matthews_corrcoef(labels, preds),
                "accuracy": accuracy_score(labels, preds),
            }

    elif task_name in ("mrpc", "qqp"):

        def fn(p) -> Any:
            """Fn."""
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            return {"accuracy": acc, "f1": f1, "accuracy_f1": (acc + f1) / 2}

    else:

        def fn(p) -> Any:
            """Fn."""
            preds = np.argmax(p.predictions, axis=1)
            labels = p.label_ids
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            }

    return fn


def aggregate_results(results_list) -> Any:
    """Aggregate results."""
    if not results_list:
        return {}
    aggregated = {}
    for key in results_list[0].keys():
        values = [r[key] for r in results_list if key in r]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_values"] = values
    return aggregated
