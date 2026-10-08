"""JSON and Parquet serialization utilities for benchmark results and trajectories."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neuralops.results_saver.storage.sqlite_ops import sanitize_json_value

logger = logging.getLogger(__name__)


def save_result_json(
    result: dict[str, Any] | Any,
    results_dir: str | Path = "07_misc_tests/benchmark_results",
    filename: str | None = None,
) -> str:
    """Save complete benchmark result including fine-grained trajectories to JSON.

    Used for generating conference-grade loss curves and variance plots.

    Args:
        result: Dictionary or object with to_dict / vars.
        results_dir: Target directory for JSON output.
        filename: Optional custom filename. If None, derived from metadata.

    Returns:
        String path to the saved JSON file.
    """
    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "to_dict") and callable(result.to_dict):
        data = result.to_dict()
    elif hasattr(result, "__dict__"):
        data = vars(result)
    else:
        data = dict(result)

    if not filename:
        mod = data.get("module_name", "unknown")
        fam = data.get("model_family", "unknown")
        ds = data.get("dataset", "unknown")
        filename = f"{mod}_{fam}_{ds}.json"

    json_path = target_dir / filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_json_value(data), f, indent=2, allow_nan=False)

    return str(json_path)


def save_trajectories_parquet(
    detailed_results: list[dict[str, Any]],
    output_dir: str | Path = "results/06_adhoc_experiments/trajectories",
) -> str | None:
    """Write step-level trajectories (loss, grad_norm, latency) as Apache Parquet.

    Falls back to JSON if pyarrow/pandas is unavailable.

    Args:
        detailed_results: List of result dictionaries containing 'trajectories'.
        output_dir: Output directory path.

    Returns:
        String path to the created parquet or json file, or None on total failure.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = target_dir / "trajectories.parquet"

    rows: list[dict[str, Any]] = []
    for entry in detailed_results:
        mod = entry.get("module_name", "unknown")
        fam = entry.get("model_family", "unknown")
        ds = entry.get("dataset", "unknown")
        run_id = entry.get("run_id", "unknown")
        trajs = entry.get("trajectories", {})
        loss_list = trajs.get("loss", [])
        grad_list = trajs.get("grad_norm", [])

        for step, loss_val in enumerate(loss_list):
            grad_val = grad_list[step] if step < len(grad_list) else None
            rows.append(
                {
                    "run_id": run_id,
                    "module_name": mod,
                    "model_family": fam,
                    "dataset": ds,
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": grad_val,
                }
            )

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(str(parquet_path), engine="pyarrow", index=False)
        return str(parquet_path)
    except Exception:
        json_fallback = target_dir / "trajectories.json"
        with open(json_fallback, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        return str(json_fallback)
