#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Central benchmark runner orchestrating config, storage, tracking, and suite execution.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from neuralops.benchmarks.core.base import ExecutionSummary, _sanitize_float
from neuralops.benchmarks.core.paths import RunDirectoryContext, StoragePathManager
from neuralops.benchmarks.suites import BENCHMARK_SUITES, get_benchmark_suite
from neuralops.config.benchmark_schema import BenchmarkRunConfig
from neuralops.tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def _get_git_provenance() -> Dict[str, Any]:
    """Retrieve Git commit SHA, branch, and working tree dirty state."""
    commit_sha = "unknown"
    branch = "unknown"
    is_dirty = False
    try:
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        )
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        )
        is_dirty = len(status) > 0
    except Exception:
        pass
    return {
        "commit_sha": commit_sha,
        "branch": branch,
        "is_dirty": is_dirty,
    }


def _get_host_environment(device_str: str) -> Dict[str, Any]:
    """Inspect and format host hardware and software environment metadata."""
    device_name = "CPU"
    compute_cap = "N/A"
    vram_mb = 0.0
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        try:
            device_name = torch.cuda.get_device_name(0)
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            compute_cap = f"{cc_major}.{cc_minor}"
            vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0), 2)
        except Exception:
            pass

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_interpreter": sys.executable,
        "pytorch_version": torch.__version__,
        "device_name": device_name,
        "compute_capability": compute_cap,
        "total_vram_mb": vram_mb,
    }


class BenchmarkRunner:
    """
    Central benchmark runner uniting:
    1. Agent 1's Config (BenchmarkRunConfig)
    2. Agent 3's Storage (StoragePathManager.resolve_run_dir)
    3. Agent 2's Tracker (ExperimentTracker)
    4. Suite Execution (AbstractBaseBenchmark)
    5. Persistence (manifest.json, summary.json, summary.csv, SQLiteMasterLedger)
    """

    def __init__(self, config: BenchmarkRunConfig) -> None:
        self.config = config
        self.paths: RunDirectoryContext = StoragePathManager.resolve_run_dir(
            benchmark_name=config.benchmark_suite,
            run_id=config.name if (config.name and "/" not in config.name) else None,
            output_root=config.output_root_directory,
        )

        # 1. Save resolved config snapshot
        self.paths.config_resolved_yaml.write_text(self.config.model_dump_yaml(), encoding="utf-8")

        # 2. Initialize tracking engine rooted in isolated per-run directory
        self.tracker = ExperimentTracker(
            log_dir=self.paths.run_dir,
            config_dict=self.config.model_dump(mode="json"),
        )

    def run(self) -> ExecutionSummary:
        """Execute the entire benchmark lifecycle from setup to ledger sync."""
        start_time_epoch = time.time()
        start_iso = datetime.now(timezone.utc).isoformat()

        git_prov = _get_git_provenance()
        host_env = _get_host_environment(self.config.hardware.device)

        manifest_data: Dict[str, Any] = {
            "$schema": "https://raw.githubusercontent.com/tahamajs/NeuralOps/main/schemas/benchmark_manifest_v1.json",
            "run_id": self.paths.run_id,
            "benchmark_name": self.config.benchmark_suite,
            "timestamp_start_utc": start_iso,
            "timestamp_end_utc": None,
            "duration_seconds": None,
            "git_provenance": git_prov,
            "host_environment": host_env,
            "model_name": getattr(self.config, "models", None) or getattr(self.config, "model", None),
            "trust_remote_code": bool(getattr(self.config, "trust_remote_code", False)),
            "execution_outcome": {
                "status": "RUNNING",
                "total_iterations": 0,
                "passed_iterations": 0,
                "failed_oom_iterations": 0,
                "failed_runtime_iterations": 0,
                "error_message": None,
            },
        }
        with open(self.paths.manifest_json, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        suite_cls = get_benchmark_suite(self.config.benchmark_suite)
        suite = suite_cls(self.config, self.tracker, self.paths)

        error_message: Optional[str] = None
        try:
            summary = suite.execute()
        except Exception as e:
            error_message = str(e)
            logger.exception("Benchmark suite '%s' failed: %s", self.config.benchmark_suite, e)
            summary = ExecutionSummary(
                run_id=self.paths.run_id,
                benchmark_name=self.config.benchmark_suite,
                status="FAILED",
                extra_metadata={"error": str(e)},
            )
        finally:
            self.tracker.close()

        end_time_epoch = time.time()
        end_iso = datetime.now(timezone.utc).isoformat()
        duration_s = _sanitize_float(end_time_epoch - start_time_epoch)
        summary.duration_seconds = duration_s

        manifest_data["timestamp_end_utc"] = end_iso
        manifest_data["duration_seconds"] = duration_s
        manifest_data["execution_outcome"] = {
            "status": summary.status,
            "total_iterations": summary.total_iterations,
            "passed_iterations": summary.passed_iterations,
            "failed_oom_iterations": summary.failed_oom_iterations,
            "failed_runtime_iterations": summary.failed_runtime_iterations,
            "error_message": error_message,
        }
        with open(self.paths.manifest_json, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        with open(self.paths.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        self._write_summary_csv(summary)
        self._write_markdown_report(summary, host_env, duration_s)

        if not self.paths.log_file.exists():
            log_msg = f"Benchmark {self.paths.run_id} completed with status: {summary.status}\n"
            self.paths.log_file.write_text(log_msg, encoding="utf-8")

        StoragePathManager.update_latest_symlink(self.paths)
        self._commit_to_master_ledger(summary, git_prov, host_env, start_iso, end_iso, duration_s)

        return summary

    def _write_summary_csv(self, summary: ExecutionSummary) -> None:
        if summary.results_table:
            fieldnames = list(summary.results_table[0].keys())
            with open(self.paths.summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in summary.results_table:
                    writer.writerow(row)
        else:
            with open(self.paths.summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["run_id", "status", "mean_latency_ms", "duration_seconds"])
                writer.writerow([summary.run_id, summary.status, summary.mean_latency_ms, summary.duration_seconds])

    def _write_markdown_report(
        self,
        summary: ExecutionSummary,
        host_env: Dict[str, Any],
        duration_s: float,
    ) -> None:
        lines = [
            f"# Benchmark Run Report: {self.config.benchmark_suite}\n",
            f"- **Run ID**: `{summary.run_id}`",
            f"- **Status**: **{summary.status}**",
            f"- **Duration**: {duration_s:.2f}s",
            f"- **Host**: `{host_env.get('hostname')}` ({host_env.get('device_name')})",
            f"- **Total Iterations**: {summary.total_iterations}",
            f"- **Passed**: {summary.passed_iterations}",
            f"- **Failed (OOM)**: {summary.failed_oom_iterations}",
            f"- **Failed (Runtime)**: {summary.failed_runtime_iterations}",
            f"- **Mean Latency**: {summary.mean_latency_ms:.3f} ms",
            f"- **P50 Latency**: {summary.p50_latency_ms:.3f} ms",
            f"- **P90 Latency**: {summary.p90_latency_ms:.3f} ms",
            f"- **P99 Latency**: {summary.p99_latency_ms:.3f} ms\n",
        ]
        self.paths.report_md.write_text("\n".join(lines), encoding="utf-8")

    def _commit_to_master_ledger(
        self,
        summary: ExecutionSummary,
        git_prov: Dict[str, Any],
        host_env: Dict[str, Any],
        start_iso: str,
        end_iso: str,
        duration_s: float,
    ) -> None:
        try:
            from neuralops.results_saver.storage.sqlite_backend import SQLiteMasterLedger

            with SQLiteMasterLedger() as ledger:
                ledger.record_benchmark_run(
                    run_id=self.paths.run_id,
                    benchmark_name=self.config.benchmark_suite,
                    commit_sha=git_prov["commit_sha"],
                    hostname=host_env["hostname"],
                    device_name=host_env["device_name"],
                    run_dir=str(self.paths.run_dir),
                    status=summary.status,
                    suite=self.config.benchmark_suite,
                    timestamp_start_utc=start_iso,
                    timestamp_end_utc=end_iso,
                    duration_seconds=duration_s,
                    branch=git_prov["branch"],
                    is_dirty=1 if git_prov["is_dirty"] else 0,
                    tag=self.config.tag,
                    total_cells=summary.total_iterations,
                    passed_cells=summary.passed_iterations,
                    failed_cells=summary.failed_oom_iterations + summary.failed_runtime_iterations,
                )
                if summary.results_table:
                    metric_rows = []
                    for row in summary.results_table:
                        entry = dict(row)
                        entry["run_id"] = self.paths.run_id
                        metric_rows.append(entry)
                    ledger.record_benchmark_metrics(metric_rows)
        except Exception as e:
            logger.warning("Failed to commit run to SQLite master ledger: %s", e)
