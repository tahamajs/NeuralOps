from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


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
                "f1_weighted": f1_score(
                    labels, preds, average="weighted", zero_division=0
                ),
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
