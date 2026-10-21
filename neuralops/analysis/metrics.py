#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Analysis Metrics Engine

Core quantitative metrics computing effective rank, spectral decay, compression ratio, and speedups.
"""

import json
import os
from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(
    predictions: np.ndarray, labels: np.ndarray, task: str = "classification"
) -> dict[str, float]:
    """Compute and return metrics."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("predictions and labels must be non-empty")
    if task not in ("classification", "regression", "stsb"):
        raise ValueError(f"Unsupported task type: {task}")

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    metrics = {}
    if task in ("regression", "stsb"):
        pred_flat = predictions.reshape(-1)
        labels_flat = labels.reshape(-1)
        if pred_flat.shape != labels_flat.shape:
            raise ValueError(
                "regression predictions and labels must contain the same number "
                f"of values, got {pred_flat.size} and {labels_flat.size}"
            )
        if pred_flat.size < 2:
            raise ValueError("regression metrics require at least two observations")
        metrics["pearson"] = pearsonr(pred_flat, labels_flat)[0]
        metrics["spearman"] = spearmanr(pred_flat, labels_flat)[0]
        metrics["pearson_spearman"] = (metrics["pearson"] + metrics["spearman"]) / 2
    else:
        preds = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
        if preds.shape != labels.shape:
            raise ValueError(
                "classification predictions and labels must have the same shape, "
                f"got {preds.shape} and {labels.shape}"
            )
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
        metrics["f1_weighted"] = f1_score(labels, preds, average="weighted", zero_division=0)
        if len(np.unique(labels)) == 2:
            metrics["matthews_corrcoef"] = matthews_corrcoef(labels, preds)
    return metrics


def aggregate_results(results_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results."""
    if not results_list:
        return {}
    aggregated = {}
    keys = dict.fromkeys(key for result in results_list for key in result)
    for key in keys:
        values = [r[key] for r in results_list if key in r and r[key] is not None]
        if values:
            try:
                numeric_values = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                continue
            aggregated[f"{key}_mean"] = np.mean(numeric_values)
            aggregated[f"{key}_std"] = np.std(numeric_values)
            aggregated[f"{key}_values"] = numeric_values.tolist()
    return aggregated


class MetricsTracker:
    """Tracks per‑step metrics and saves as JSON lines."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.records: list[dict[str, Any]] = []
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def log(self, metrics: dict[str, Any], step: int | None = None, flush: bool = False) -> None:
        """Log."""
        entry = {"step": step} if step is not None else {}
        entry.update(metrics)
        self.records.append(entry)
        if flush:
            self._flush()

    def _flush(self) -> None:
        """flush."""
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
