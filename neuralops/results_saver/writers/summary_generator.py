#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multi-Format Benchmark Summary Generator

Computes descriptive aggregates (mean, std, median, min, max, P90, P99) and
generates summary.json, summary.csv, and benchmark_report.md.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BenchmarkSummaryGenerator:
    """Computes descriptive statistics and serializes run summary artifacts."""

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Compute percentile p (0-100) using linear interpolation."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        if p <= 0:
            return sorted_vals[0]
        if p >= 100:
            return sorted_vals[-1]
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return d0 + d1

    @classmethod
    def compute_aggregates(cls, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate statistics across benchmark iteration records."""
        if not iterations:
            return {
                "total_iterations": 0,
                "passed_iterations": 0,
                "failed_runtime_iterations": 0,
                "latency_mean_ms": 0.0,
                "latency_std_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_median_ms": 0.0,
                "latency_p90_ms": 0.0,
                "latency_p99_ms": 0.0,
                "latency_min_ms": 0.0,
                "latency_max_ms": 0.0,
                "peak_vram_mb": 0.0,
                "peak_memory_mb": 0.0,
            }

        latencies: List[float] = []
        memories: List[float] = []
        failed = 0

        for it in iterations:
            status = it.get("status", "PASSED")
            if status != "PASSED":
                failed += 1
            lat = it.get("latency_ms") or it.get("forward_latency_ms") or it.get("mean_latency_ms")
            if lat is not None and isinstance(lat, (int, float)) and not math.isnan(lat):
                latencies.append(float(lat))
            mem = (
                it.get("peak_vram_mb")
                or it.get("vram_mb")
                or it.get("peak_memory_mb")
                or it.get("memory_mb")
                or it.get("gpu_memory_allocated_mb")
            )
            if mem is not None and isinstance(mem, (int, float)) and not math.isnan(mem):
                memories.append(float(mem))

        n = len(latencies)
        mean_lat = sum(latencies) / n if n > 0 else 0.0
        variance = sum((x - mean_lat) ** 2 for x in latencies) / n if n > 1 else 0.0
        std_lat = math.sqrt(variance)
        p50_val = round(cls._percentile(latencies, 50.0), 4) if latencies else 0.0
        peak_mem = round(max(memories), 2) if memories else 0.0

        return {
            "total_iterations": len(iterations),
            "passed_iterations": len(iterations) - failed,
            "failed_runtime_iterations": failed,
            "latency_mean_ms": round(mean_lat, 4),
            "latency_std_ms": round(std_lat, 4),
            "latency_p50_ms": p50_val,
            "latency_median_ms": p50_val,
            "latency_p90_ms": round(cls._percentile(latencies, 90.0), 4) if latencies else 0.0,
            "latency_p99_ms": round(cls._percentile(latencies, 99.0), 4) if latencies else 0.0,
            "latency_min_ms": round(min(latencies), 4) if latencies else 0.0,
            "latency_max_ms": round(max(latencies), 4) if latencies else 0.0,
            "peak_vram_mb": peak_mem,
            "peak_memory_mb": peak_mem,
        }

    @classmethod
    def generate_summary_json(
        cls,
        run_id: str,
        benchmark_name: str,
        aggregates: Dict[str, Any],
        manifest: Optional[Dict[str, Any]] = None,
        iterations: Optional[List[Dict[str, Any]]] = None,
        target_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Serialize run summary to JSON."""
        iterations = iterations or []
        payload = {
            "run_id": run_id,
            "benchmark_name": benchmark_name,
            "aggregates": aggregates,
            "manifest": manifest or {},
            "total_iterations": len(iterations),
        }

        if target_path is not None:
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)

        return payload
            "manifest": manifest or {},
            "total_iterations": len(iterations),
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return target

    @classmethod
    def generate_summary_csv(
        cls,
        iterations: List[Dict[str, Any]],
        target_path: Path | str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Serialize iterations to CSV."""
        if not iterations:
            return None
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        extra = extra_fields or {}
        # Collect all keys
        keys_set = set(extra.keys())
        for it in iterations:
            keys_set.update(it.keys())
        fieldnames = sorted(list(keys_set))

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for it in iterations:
                row = {**extra, **it}
                # format floats to 4 decimals
                for k, v in row.items():
                    if isinstance(v, float):
                        row[k] = round(v, 4)
                writer.writerow(row)
        return target

    @classmethod
    def generate_markdown_report(
        cls,
        run_id: str,
        benchmark_name: str,
        aggregates: Dict[str, Any],
        manifest: Optional[Dict[str, Any]],
        iterations: List[Dict[str, Any]],
        target_path: Path | str,
    ) -> Path:
        """Generate GitHub-flavored Markdown report."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        device_name = "Unknown"
        if manifest and "hardware" in manifest:
            device_name = manifest["hardware"].get("device_name", "Unknown")

        lines = [
            f"# Benchmark Report: {benchmark_name}",
            "",
            f"- **Run ID**: `{run_id}`",
            f"- **Device**: `{device_name}`",
            f"- **Total Iterations**: {aggregates.get('total_iterations', 0)}",
            f"- **Passed**: {aggregates.get('passed_iterations', 0)}",
            f"- **Failed**: {aggregates.get('failed_runtime_iterations', 0)}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| Latency Mean (ms) | {aggregates.get('latency_mean_ms', 0.0):.4f} |",
            f"| Latency Median (ms) | {aggregates.get('latency_median_ms', 0.0):.4f} |",
            f"| Latency P90 (ms) | {aggregates.get('latency_p90_ms', 0.0):.4f} |",
            f"| Latency P99 (ms) | {aggregates.get('latency_p99_ms', 0.0):.4f} |",
            f"| Latency Min / Max (ms) | {aggregates.get('latency_min_ms', 0.0):.4f} / {aggregates.get('latency_max_ms', 0.0):.4f} |",
            f"| Peak Memory (MB) | {aggregates.get('peak_memory_mb', 0.0):.2f} |",
            "",
        ]

        if iterations:
            lines.append("## Iteration Samples (First 10)")
            lines.append("")
            # Sample up to 10
            sample = iterations[:10]
            cols = ["iteration", "batch_size", "seq_len", "latency_ms", "status"]
            # Filter cols to existing keys
            valid_cols = [c for c in cols if any(c in row for row in sample)]
            if valid_cols:
                header = "| " + " | ".join(valid_cols) + " |"
                sep = "| " + " | ".join([":---"] * len(valid_cols)) + " |"
                lines.append(header)
                lines.append(sep)
                for row in sample:
                    row_vals = [str(round(row[c], 4) if isinstance(row.get(c), float) else row.get(c, "-")) for c in valid_cols]
                    lines.append("| " + " | ".join(row_vals) + " |")
                lines.append("")

        with open(target, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return target
