"""Metrics computation, aggregation, and tracking."""

import json
import os
from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(
    predictions: np.ndarray, labels: np.ndarray, task: str = "classification"
) -> dict[str, float]:
    """Compute and return metrics."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    metrics = {}
    if task in ("regression", "stsb"):
        pred_flat = np.squeeze(predictions)
        labels_flat = np.squeeze(labels)
        metrics["pearson"] = pearsonr(pred_flat, labels_flat)[0]
        metrics["spearman"] = spearmanr(pred_flat, labels_flat)[0]
        metrics["pearson_spearman"] = (metrics["pearson"] + metrics["spearman"]) / 2
    else:
        preds = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
        metrics["f1_weighted"] = f1_score(
            labels, preds, average="weighted", zero_division=0
        )
        if len(np.unique(labels)) == 2:
            metrics["matthews_corrcoef"] = matthews_corrcoef(labels, preds)
    return metrics


def aggregate_results(results_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results."""
    if not results_list:
        return {}
    aggregated = {}
    for key in results_list[0].keys():
        values = [r[key] for r in results_list if key in r and r[key] is not None]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_values"] = values
    return aggregated


class MetricsTracker:
    """Tracks per‑step metrics and saves as JSON lines."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.records: list[dict[str, Any]] = []
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(
        self, metrics: dict[str, Any], step: int | None = None, flush: bool = False
    ) -> None:
        """Log."""
        entry = {"step": step} if step is not None else {}
        entry.update(metrics)
        self.records.append(entry)
        if flush:
            self._flush()

    def _flush(self) -> None:
        """ flush."""
        if self.filepath:
            with open(self.filepath, "a") as f:
                for rec in self.records:
                    f.write(json.dumps(rec) + "\n")
            self.records.clear()

    def save(self) -> None:
        """Save."""
        self._flush()

    def to_dataframe(self) -> pd.DataFrame:
        """To dataframe."""
        all_records = []
        if os.path.exists(self.filepath):
            with open(self.filepath) as f:
                for line in f:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        all_records.extend(self.records)
        return pd.DataFrame(all_records)
