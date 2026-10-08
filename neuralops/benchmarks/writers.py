"""Comprehensive writers and persistence utilities for benchmark results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from neuralops.benchmarks.result import BenchmarkResult


def write_json(result: BenchmarkResult, path: Path | str) -> Path:
    """Save single BenchmarkResult to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fp:
        json.dump(result.to_dict(), fp, indent=2)
    return p


def write_csv(results: list[BenchmarkResult], path: Path | str) -> Path:
    """Flatten and write list of BenchmarkResults to summary CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return p

    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "run_id": r.run_id,
            "status": r.status,
            "module": r.module.get("name", "") if isinstance(r.module, dict) else str(r.module),
            "module_type": r.module.get("type", "") if isinstance(r.module, dict) else "",
            "model": r.model.get("name", "") if isinstance(r.model, dict) else str(r.model),
            "dataset": r.dataset.get("name", "") if isinstance(r.dataset, dict) else str(r.dataset),
            "rank": r.config.get("rank", 0) if isinstance(r.config, dict) else 0,
            "timed_out": r.timed_out,
            "error": r.error or "",
        }
        if r.metrics:
            for k, v in r.metrics.items():
                row[f"metric_{k}"] = v
        if r.resources:
            for k, v in r.resources.items():
                row[f"resource_{k}"] = v
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p


def write_markdown(results: list[BenchmarkResult], path: Path | str) -> Path:
    """Write human-readable GitHub-flavored Markdown table summary."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Execution Summary",
        f"Total runs: {len(results)}",
        "",
        "| # | Module | Type | Model | Dataset | Rank | Status | Acc / PPL | Latency (ms) | Peak VRAM (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for idx, r in enumerate(results, 1):
        mod_name = r.module.get("name", "unknown") if isinstance(r.module, dict) else str(r.module)
        mod_type = r.module.get("type", "unknown") if isinstance(r.module, dict) else ""
        model_name = r.model.get("name", "unknown") if isinstance(r.model, dict) else str(r.model)
        dataset_name = r.dataset.get("name", "unknown") if isinstance(r.dataset, dict) else str(r.dataset)
        rank_val = r.config.get("rank", "-") if isinstance(r.config, dict) else "-"

        metric_str = "N/A"
        if r.metrics:
            if "accuracy" in r.metrics and r.metrics["accuracy"] is not None:
                metric_str = f"{r.metrics['accuracy']*100:.2f}%"
            elif "perplexity" in r.metrics and r.metrics["perplexity"] is not None:
                metric_str = f"PPL {r.metrics['perplexity']:.2f}"

        lat_str = "N/A"
        vram_str = "N/A"
        if r.resources:
            if "latency_ms_per_token" in r.resources:
                lat_str = f"{r.resources['latency_ms_per_token']:.3f}"
            if "peak_vram_mb" in r.resources:
                vram_str = f"{r.resources['peak_vram_mb']:.1f}"

        status_badge = f"**{r.status.upper()}**" if r.status == "success" else f"`{r.status}`"
        lines.append(
            f"| {idx} | `{mod_name}` | {mod_type} | `{model_name}` | `{dataset_name}` | {rank_val} | {status_badge} | {metric_str} | {lat_str} | {vram_str} |"
        )

    lines.append("")
    with open(p, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    return p


def write_all(result: BenchmarkResult, base_dir: Path | str) -> Path:
    """Write run result according to canonical hierarchy."""
    b = Path(base_dir)
    mod_name = result.module.get("name", "module") if isinstance(result.module, dict) else str(result.module)
    model_name = Path(result.model.get("name", "model") if isinstance(result.model, dict) else str(result.model)).name.replace("-", "_")
    dataset_name = result.dataset.get("name", "dataset") if isinstance(result.dataset, dict) else str(result.dataset)

    target_dir = b / "runs" / dataset_name / model_name / mod_name
    target_path = target_dir / f"{result.run_id}.json"
    write_json(result, target_path)
    return target_path


__all__ = [
    "write_json",
    "write_csv",
    "write_markdown",
    "write_all",
]
