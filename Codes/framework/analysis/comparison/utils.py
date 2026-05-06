"""Utility functions to load results from various sources."""

import json, os, pandas as pd, numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_results_from_directories(
    dirs: List[str],
    group_names: Optional[List[str]] = None,
    metric: str = "eval_accuracy",
) -> Dict[str, np.ndarray]:
    """Read final_metrics.json from multiple run directories and collect scores by group."""
    groups: Dict[str, List[float]] = {}
    for i, d in enumerate(dirs):
        name = group_names[i] if group_names else Path(d).name
        path = Path(d) / "artifacts" / "final_metrics.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        score = data.get(metric)
        if score is not None:
            groups.setdefault(name, []).append(float(score))
    return {k: np.array(v) for k, v in groups.items()}


def load_results_from_django(
    runs,
    metric: str = "final_metrics__eval_accuracy",
    group_by: str = "config__attn_type",
) -> pd.DataFrame:
    """
    Build a DataFrame from Django TrainingRun queryset.
    `runs` is a queryset with `final_metrics` and `config` fields.
    """
    records = []
    for run in runs:
        metrics = run.final_metrics or {}
        config = run.config or {}
        value = metrics.get(metric.split('__')[-1], None)
        if value is not None:
            records.append({
                "run_id": str(run.id),
                "group": config.get(group_by.split('__')[-1], "unknown"),
                metric: float(value),
            })
    return pd.DataFrame(records)


def load_results_from_glue_results(
    csv_path: str,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Load a CSV file with columns: model, task, rank, accuracy, etc."""
    return pd.read_csv(csv_path)