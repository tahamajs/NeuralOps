#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Rolling Suite Aggregator

Maintains a canonical rolling master summary CSV directly in the suite root:
results/benchmarks/<suite>/summary.csv
Appends historical execution rows across commits with atomic filesystem guarantees.
"""

from __future__ import annotations

import csv
import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ROLLING_DEFAULT_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "benchmark_name",
    "commit_sha",
    "device_name",
    "status",
    "duration_seconds",
    "total_iterations",
    "passed_iterations",
    "failed_oom_iterations",
    "failed_runtime_iterations",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p99_ms",
    "peak_vram_mb",
    "mean_tflops",
]


class RollingSuiteAggregator:
    """Safely appends benchmark run summaries to suite-level rolling CSV tables."""

    @classmethod
    def append_run(
        cls,
        suite_dir: Path | str,
        run_id: str,
        benchmark_name: str,
        aggregates: Dict[str, Any],
        manifest: Optional[Dict[str, Any]] = None,
        duration_seconds: float = 0.0,
        status: str = "PASSED",
    ) -> Path:
        """
        Atomically append a run's summary to <suite_dir>/summary.csv.
        """
        suite_path = Path(suite_dir)
        suite_path.mkdir(parents=True, exist_ok=True)
        master_csv = suite_path / "summary.csv"

        manifest = manifest or {}
        commit_sha = (manifest.get("git_provenance") or {}).get("commit_sha", "unknown")
        device_name = manifest.get("device_name") or "unknown"
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        row: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": now_utc,
            "benchmark_name": benchmark_name,
            "commit_sha": commit_sha,
            "device_name": device_name,
            "status": status,
            "duration_seconds": round(duration_seconds, 4),
            "total_iterations": aggregates.get("total_iterations", 0),
            "passed_iterations": aggregates.get("passed_iterations", 0),
            "failed_oom_iterations": aggregates.get("failed_oom_iterations", 0),
            "failed_runtime_iterations": aggregates.get("failed_runtime_iterations", 0),
            "latency_mean_ms": aggregates.get("latency_mean_ms", 0.0),
            "latency_p50_ms": aggregates.get("latency_p50_ms", 0.0),
            "latency_p90_ms": aggregates.get("latency_p90_ms", 0.0),
            "latency_p99_ms": aggregates.get("latency_p99_ms", 0.0),
            "peak_vram_mb": aggregates.get("peak_vram_mb", 0.0),
            "mean_tflops": aggregates.get("mean_tflops", 0.0),
        }

        # Read existing rows to prevent schema drift or duplicate headers
        existing_rows: List[Dict[str, Any]] = []
        headers = list(ROLLING_DEFAULT_COLUMNS)

        if master_csv.exists() and master_csv.stat().st_size > 0:
            try:
                with open(master_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        headers = list(reader.fieldnames)
                    for r in reader:
                        existing_rows.append(r)
            except Exception:
                existing_rows = []

        # Merge headers if new keys exist
        for k in row.keys():
            if k not in headers:
                headers.append(k)

        # Write to temporary file for atomic replacement
        temp_csv = suite_path / f".tmp_summary_{run_id}.csv"
        try:
            with open(temp_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)
                writer.writerow(row)

            os.replace(temp_csv, master_csv)
        finally:
            if temp_csv.exists():
                try:
                    temp_csv.unlink()
                except OSError:
                    pass

        return master_csv
