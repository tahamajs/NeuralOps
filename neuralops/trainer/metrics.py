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


def _prepare_inputs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 1D squeezed numpy arrays and validate shape compatibility."""
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    t_arr = np.asarray(y_true).squeeze()
    p_arr = np.asarray(y_pred).squeeze()

    # Handle 0-dim scalar case
    if t_arr.ndim == 0 and p_arr.ndim == 0:
        t_arr = np.atleast_1d(t_arr)
        p_arr = np.atleast_1d(p_arr)

    if t_arr.shape != p_arr.shape:
        raise ValueError(f"Shape mismatch: y_true shape {t_arr.shape} does not match y_pred shape {p_arr.shape}")
    return t_arr, p_arr


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, label: Any) -> tuple[int, int, int]:
    """Return (TP, FP, FN) for a single label."""
    tp = int(np.sum((y_pred == label) & (y_true == label)))
    fp = int(np.sum((y_pred == label) & (y_true != label)))
    fn = int(np.sum((y_pred != label) & (y_true == label)))
    return tp, fp, fn


def accuracy_score(y_true: Any, y_pred: Any) -> float:
    """Fraction of correctly classified samples.

    Strictly validates input shapes, preventing unintended outer-product broadcasting
    when given (N, 1) and (N,) inputs.
    """
    t_arr, p_arr = _prepare_inputs(y_true, y_pred)
    if t_arr.size == 0:
        return 0.0
    return float(np.mean(t_arr == p_arr))


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Mean squared error."""
    t_arr, p_arr = _prepare_inputs(y_true, y_pred)
    if t_arr.size == 0:
        return 0.0
    t_float = t_arr.astype(float)
    p_float = p_arr.astype(float)
    return float(np.mean((t_float - p_float) ** 2))


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error."""
    t_arr, p_arr = _prepare_inputs(y_true, y_pred)
    if t_arr.size == 0:
        return 0.0
    t_float = t_arr.astype(float)
    p_float = p_arr.astype(float)
    return float(np.mean(np.abs(t_float - p_float)))


def f1_score(y_true: Any, y_pred: Any, average: str = "weighted", zero_division: int = 0) -> Any:
    """F1 score (macro / weighted / micro / None)."""
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=zero_division)
    return f1


def matthews_corrcoef(y_true: Any, y_pred: Any) -> float:
    """Matthews correlation coefficient (supports binary and multiclass classification)."""
    t_arr, p_arr = _prepare_inputs(y_true, y_pred)
    if t_arr.size == 0:
        return 0.0
    labels = np.unique(np.concatenate([t_arr, p_arr]))
    if len(labels) <= 1:
        return 0.0

    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    num_classes = len(labels)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t_val, p_val in zip(t_arr, p_arr):
        conf_mat[label_to_idx[t_val], label_to_idx[p_val]] += 1

    t_k = conf_mat.sum(axis=1).astype(float)
    p_k = conf_mat.sum(axis=0).astype(float)
    c = float(np.trace(conf_mat))
    s = float(t_arr.size)
    numerator = c * s - float(np.dot(p_k, t_k))
    denominator = float(np.sqrt((s**2 - np.sum(p_k**2)) * (s**2 - np.sum(t_k**2))))
    return float(numerator / denominator) if denominator > 0 else 0.0


def precision_recall_fscore_support(
    y_true: Any, y_pred: Any, average: str | None = "weighted", zero_division: int = 0
) -> tuple:
    """Precision, recall, F-score, and support matching sklearn reference."""
    t_arr, p_arr = _prepare_inputs(y_true, y_pred)
    if t_arr.size == 0:
        return (0.0, 0.0, 0.0, None) if average is not None else ([], [], [], [])

    labels = np.unique(np.concatenate([t_arr, p_arr]))
    ps, rs, f1s, supports = [], [], [], []
    for lbl in labels:
        tp, fp, fn = _confusion(t_arr, p_arr, lbl)
        p = tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)
        r = tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)
        denom = 2 * tp + fp + fn
        f = (2 * tp / denom) if denom > 0 else float(zero_division)
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
        total_w = w.sum()
        return (
            float(np.average(ps_arr, weights=w)) if total_w > 0 else float(zero_division),
            float(np.average(rs_arr, weights=w)) if total_w > 0 else float(zero_division),
            float(np.average(f1s_arr, weights=w)) if total_w > 0 else float(zero_division),
            None,
        )
    if average == "macro":
        return float(np.mean(ps_arr)), float(np.mean(rs_arr)), float(np.mean(f1s_arr)), None
    if average == "micro":
        total_tp = sum(int(np.sum((p_arr == lbl) & (t_arr == lbl))) for lbl in labels)
        total_fp = sum(int(np.sum((p_arr == lbl) & (t_arr != lbl))) for lbl in labels)
        total_fn = sum(int(np.sum((p_arr != lbl) & (t_arr == lbl))) for lbl in labels)
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float(zero_division)
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float(zero_division)
        micro_denom = 2 * total_tp + total_fp + total_fn
        micro_f = (2 * total_tp / micro_denom) if micro_denom > 0 else float(zero_division)
        return float(micro_p), float(micro_r), float(micro_f), None
    return ps_arr.tolist(), rs_arr.tolist(), f1s_arr.tolist(), sup_arr.tolist()


def precision_score(y_true: Any, y_pred: Any, average: str = "weighted", zero_division: int = 0) -> float:
    """Weighted/macro precision score."""
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=zero_division)
    return float(p)


def recall_score(y_true: Any, y_pred: Any, average: str = "weighted", zero_division: int = 0) -> float:
    """Weighted/macro recall score."""
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=zero_division)
    return float(r)


def compute_perplexity(loss: float | Any) -> float:
    """Compute perplexity metric from cross-entropy loss value."""
    if hasattr(loss, "item"):
        val = float(loss.item())
    else:
        val = float(loss)
    return float(np.exp(val))


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
    p, r, _, _ = precision_recall_fscore_support(labels, predictions, average=average, zero_division=0)
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
