"""Standardized result writers and schema for neuralops benchmarks."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid


@dataclass
class BenchmarkResult:
    """Canonical schema for benchmark execution result."""

    benchmark: str
    module: str
    model: str
    dataset: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    resources: dict[str, Any]
    status: str = "PASS"
    duration_s: float = 0.0
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: str = "unknown"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create result from dictionary."""
        return cls(**data)


def write_json(result: BenchmarkResult, path: Path | str) -> Path:
    """Write benchmark result to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fp:
        json.dump(result.to_dict(), fp, indent=2)
    return p


def write_csv(results: list[BenchmarkResult], path: Path | str) -> Path:
    """Write list of benchmark results to a flat summary CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return p

    # Flatten nested fields
    flattened_rows = []
    for r in results:
        row: dict[str, Any] = {
            "run_id": r.run_id,
            "timestamp": r.timestamp,
            "benchmark": r.benchmark,
            "module": r.module,
            "model": r.model,
            "dataset": r.dataset,
            "status": r.status,
            "duration_s": r.duration_s,
            "git_commit": r.git_commit,
            "error": r.error or "",
        }
        for k, v in r.config.items():
            row[f"config_{k}"] = v
        for k, v in r.metrics.items():
            row[f"metric_{k}"] = v
        for k, v in r.resources.items():
            row[f"resource_{k}"] = v
        flattened_rows.append(row)

    all_keys = list(flattened_rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(flattened_rows)
    return p


def write_markdown(results: list[BenchmarkResult], path: Path | str) -> Path:
    """Write formatted markdown summary table."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Benchmark Results Summary: {results[0].benchmark if results else 'Run'}",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Module | Model | Dataset | Status | Primary Metric | Latency (ms) | Peak VRAM (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        primary_metric = (
            f"Acc: {r.metrics.get('accuracy_pct', 'N/A')}%"
            if "accuracy_pct" in r.metrics
            else f"PPL: {r.metrics.get('perplexity', 'N/A')}"
        )
        lat = r.resources.get("latency_ms", "N/A")
        vram = r.resources.get("peak_vram_mb", "N/A")
        lines.append(
            f"| `{r.module}` | `{r.model}` | `{r.dataset}` | **{r.status}** | {primary_metric} | {lat} | {vram} |"
        )
    lines.append("")
    with open(p, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    return p


def write_all(result: BenchmarkResult, base_dir: Path | str) -> Path:
    """
    Write result using the canonical standard hierarchy:
    results/<benchmark>/<dataset>/<model>/<module>/result.json
    """
    b = Path(base_dir)
    safe_model = Path(result.model).name.replace("-", "_").lower()
    target_dir = b / result.benchmark / result.dataset / safe_model / result.module
    target_path = target_dir / "result.json"
    write_json(result, target_path)
    return target_path
