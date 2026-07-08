"""
Advanced analytics: hyperparameter importance, correlation matrix,
learning curve analysis, and statistical comparison.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .models import TrainingRun, TrainingSnapshot


def hyperparameter_importance(
    run_ids: Optional[List[str]] = None,
    target_metric: str = "eval_accuracy",
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Estimate hyperparameter importance by correlating each HP
    with the target metric across runs. Uses Pearson correlation.
    """
    qs = TrainingRun.objects.filter(
        status="completed", final_metrics__has_key=target_metric
    )
    if run_ids:
        qs = qs.filter(id__in=run_ids)
    runs = list(qs)

    hp_keys = [
        "learning_rate",
        "batch_size",
        "epochs",
        "max_seq_length",
        "compression_rank",
        "lora_r",
        "lora_alpha",
        "weight_decay",
        "warmup_steps",
        "grad_accum",
    ]
    hp_values = defaultdict(list)
    metric_values = []

    for run in runs:
        cfg = run.config or {}
        m = (run.final_metrics or {}).get(target_metric)
        if m is None:
            continue
        metric_values.append(m)
        for k in hp_keys:
            v = cfg.get(k)
            hp_values[k].append(v if v is not None else np.nan)

    if len(metric_values) < 3:
        return {"error": "Need at least 3 completed runs with the target metric"}

    metric_arr = np.array(metric_values, dtype=float)
    importances = []
    for k in hp_keys:
        vals = np.array(hp_values[k], dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            continue
        corr = np.corrcoef(vals[mask], metric_arr[mask])[0, 1]
        importances.append(
            {
                "parameter": k,
                "correlation": round(float(corr), 4),
                "abs_correlation": round(abs(float(corr)), 4),
            }
        )

    importances.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return {
        "target_metric": target_metric,
        "num_runs": len(metric_values),
        "importances": importances[:top_k],
    }


def metric_correlation_matrix(run_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build correlation matrix between all metrics across runs."""
    qs = TrainingRun.objects.filter(status="completed")
    if run_ids:
        qs = qs.filter(id__in=run_ids)
    runs = list(qs)

    all_metric_names = set()
    rows = []
    for run in runs:
        m = run.final_metrics or {}
        all_metric_names.update(m.keys())
        rows.append(m)

    metric_names = sorted(all_metric_names)
    if len(metric_names) < 2:
        return {"error": "Need at least 2 metrics"}

    matrix = np.full((len(metric_names), len(metric_names)), np.nan)
    for i, mi in enumerate(metric_names):
        for j, mj in enumerate(metric_names):
            vals_i = [
                r.get(mi)
                for r in rows
                if r.get(mi) is not None and r.get(mj) is not None
            ]
            vals_j = [
                r.get(mj)
                for r in rows
                if r.get(mi) is not None and r.get(mj) is not None
            ]
            if len(vals_i) >= 3:
                matrix[i][j] = np.corrcoef(vals_i, vals_j)[0, 1]

    return {
        "metrics": metric_names,
        "correlation_matrix": [
            [float(v) if not np.isnan(v) else None for v in row] for row in matrix
        ],
        "num_runs": len(runs),
    }


def training_stability_analysis(run_id: str) -> Dict[str, Any]:
    """Analyze training stability from snapshot history."""
    snapshots = TrainingSnapshot.objects.filter(run_id=run_id).order_by("step")
    if not snapshots.exists():
        return {"error": "No snapshots found"}

    losses = [s.train_loss for s in snapshots if s.train_loss is not None]
    if len(losses) < 10:
        return {"error": "Need at least 10 snapshots"}

    loss_arr = np.array(losses)
    window = max(len(losses) // 10, 5)

    # Rolling statistics
    rolling_mean = np.convolve(loss_arr, np.ones(window) / window, mode="valid")
    rolling_std = np.array(
        [np.std(loss_arr[max(0, i - window) : i + 1]) for i in range(len(loss_arr))]
    )

    # Convergence detection
    last_20 = loss_arr[-20:] if len(loss_arr) >= 20 else loss_arr
    converged = (
        np.std(last_20) < 0.01 * np.mean(last_20) if np.mean(last_20) != 0 else False
    )

    return {
        "final_loss": float(loss_arr[-1]),
        "min_loss": float(loss_arr.min()),
        "loss_std_last_20": float(np.std(last_20)),
        "converged": bool(converged),
        "rolling_mean_final": (
            float(rolling_mean[-1]) if len(rolling_mean) > 0 else None
        ),
        "noise_estimate": float(np.std(loss_arr[-window:])),
        "total_steps": len(loss_arr),
    }
