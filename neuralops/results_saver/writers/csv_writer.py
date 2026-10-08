"""CSV export utilities and schema definitions for benchmark results."""

from __future__ import annotations

import csv
from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MASTER_CSV_COLUMNS: list[str] = [
    "timestamp",
    "track",
    "module_name",
    "module_type",
    "model_family",
    "dataset",
    "status",
    "error_message",
    "forward_latency_ms",
    "backward_latency_ms",
    "throughput_samples_per_sec",
    "speedup_vs_baseline",
    "param_count",
    "trainable_param_count",
    "compression_ratio",
    "peak_memory_mb",
    "loss",
    "accuracy_pct",
    "f1_score",
    "mcc",
    "pearson",
    "spearman",
    "fidelity_score",
    "flops_macs",
    "throughput_tokens_sec",
    "latency_variance_ms",
    "perplexity",
    "grad_norm",
    "ci_lower",
    "ci_upper",
    "p_value_vs_baseline",
    "mean_seed_acc",
    "std_seed_acc",
]


def save_result(
    result: dict[str, Any] | Any,
    results_dir: str | Path = "07_misc_tests/benchmark_results",
    csv_filename: str = "master_results.csv",
) -> str:
    """Append a benchmark result row to the master CSV file.

    1. Master CSV is located at results_dir / csv_filename.
    2. Header is written only if file does not exist.
    3. Each call appends one row using csv.DictWriter with extrasaction='ignore'.
    4. All float fields are rounded to 4 decimal places before writing.

    Args:
        result: Dictionary or object with to_dict / vars attributes.
        results_dir: Directory path where CSV is stored.
        csv_filename: Filename of the target CSV.

    Returns:
        Absolute string path to the updated CSV file.
    """
    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / csv_filename

    if hasattr(result, "to_dict") and callable(result.to_dict):
        row = dict(result.to_dict())
    elif hasattr(result, "__dict__"):
        row = dict(vars(result))
    elif isinstance(result, dict):
        row = dict(result)
    else:
        row = dict(result)

    if "timestamp" not in row or not row["timestamp"]:
        row["timestamp"] = datetime.now().isoformat()

    # Format fields: round floats to 4 decimal places, None to empty string
    formatted_row: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, float):
            formatted_row[k] = round(v, 4) if math.isfinite(v) else ""
        elif v is None:
            formatted_row[k] = ""
        else:
            formatted_row[k] = v

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=MASTER_CSV_COLUMNS,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(formatted_row)

    return str(csv_path)
