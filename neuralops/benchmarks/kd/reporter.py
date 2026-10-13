"""Result reporting, streaming CSV persistence, and Markdown generation.

This module provides the :class:`KDReporter` class to stream benchmark results
safely to disk using atomic file writing techniques, preventing corruption in case
of sudden termination.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuralops.benchmarks.kd.config import BenchmarkConfig


class KDReporter:
    """Reporter and file manager for KD benchmarking runs.

    Handles atomic output generation for single-run JSON files, continuous
    CSV streaming logs, aggregate JSON summaries, and Markdown leaderboards.

    Parameters
    ----------
    config : BenchmarkConfig
        Active benchmark configuration.

    Examples
    --------
    >>> from neuralops.benchmarks.kd.config import BenchmarkConfig
    >>> from neuralops.benchmarks.kd.reporter import KDReporter
    >>> cfg = BenchmarkConfig(output_dir="./test_results")
    >>> reporter = KDReporter(cfg)
    >>> reporter.record_run({"module": "sink_attention", "status": "IMPROVED"})
    """

    CSV_FIELDS = [
        "module",
        "dataset",
        "status",
        "pre_kd_acc",
        "best_kd_acc",
        "best_step",
        "final_kd_acc",
        "delta_vs_pre",
        "duration_s",
        "error",
        "timestamp",
    ]

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.runs_dir = self.output_dir / "runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "kd_benchmark.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        """Initialize the CSV header if the file does not already exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def record_run(self, result: Dict[str, Any]) -> None:
        """Record a single module evaluation atomically to disk.

        Writes a dedicated JSON file in ``runs/`` and appends the result to
        the main streaming CSV file.

        Parameters
        ----------
        result : dict of str to Any
            Metrics and metadata for the evaluated module.
        """
        module_name = result.get("module", "unknown")
        dataset_name = result.get("dataset", "default")
        result.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

        # 1. Write per-run atomic JSON
        run_file = self.runs_dir / f"{dataset_name}__{module_name}.json"
        self._atomic_write_json(run_file, result)

        # 2. Append to streaming CSV
        row = {k: result.get(k, "") for k in self.CSV_FIELDS}
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writerow(row)

    def generate_final_report(self, all_results: List[Dict[str, Any]]) -> None:
        """Generate final aggregate JSON and Markdown reports.

        Parameters
        ----------
        all_results : list of dict
            Complete collection of results for all evaluated modules.
        """
        json_path = self.output_dir / "kd_benchmark_final.json"
        md_path = self.output_dir / "kd_benchmark_final.md"

        # Save complete JSON
        self._atomic_write_json(json_path, all_results)

        # Build Markdown table and summary
        lines = [
            f"# Knowledge Distillation (KD) Benchmark Final Report",
            f"",
            f"- **Task**: `{self.config.task}`",
            f"- **Teacher**: `{self.config.teacher}`",
            f"- **Total Modules**: {len(all_results)}",
            f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"",
            f"## Module Results Summary",
            f"",
            f"| Module | Status | Pre-KD Acc (%) | Best KD Acc (%) | Post-KD (Full) | Delta (%) | Duration (s) |",
            f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        # Sort: successfully improved/evaluated first, descending by final accuracy
        def sort_key(r: Dict[str, Any]) -> float:
            val = r.get("final_kd_acc")
            return float(val) if val is not None and val != "" else -1.0

        sorted_res = sorted(all_results, key=sort_key, reverse=True)

        for r in sorted_res:
            m = r.get("module", "")
            st = r.get("status", "")
            pre = f"{r.get('pre_kd_acc', ''):.2f}" if r.get("pre_kd_acc") is not None else "N/A"
            best = f"{r.get('best_kd_acc', ''):.2f}" if r.get("best_kd_acc") is not None else "N/A"
            post = f"{r.get('final_kd_acc', ''):.2f}" if r.get("final_kd_acc") is not None else "N/A"
            delta = f"{r.get('delta_vs_pre', ''):+.2f}" if r.get("delta_vs_pre") is not None else "N/A"
            dur = f"{r.get('duration_s', ''):.1f}" if r.get("duration_s") is not None else ""
            lines.append(f"| `{m}` | **{st}** | {pre} | {best} | {post} | {delta} | {dur} |")

        lines.append("")
        self._atomic_write_text(md_path, "\n".join(lines))

    @staticmethod
    def _atomic_write_json(target_path: Path, data: Any) -> None:
        """Write JSON atomically using temporary file swap."""
        temp_file = target_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, target_path)

    @staticmethod
    def _atomic_write_text(target_path: Path, content: str) -> None:
        """Write text content atomically using temporary file swap."""
        temp_file = target_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, target_path)
