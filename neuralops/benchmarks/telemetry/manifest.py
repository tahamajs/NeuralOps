#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Execution Manifest Builder

Generates canonical provenance documents (manifest.json) conforming to
Section 4 & Appendix B of the Unified Benchmark Architecture Specification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .host import get_host_environment

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_URL = (
    "https://raw.githubusercontent.com/tahamajs/NeuralOps/main/schemas/benchmark_manifest_v1.json"
)


def get_git_provenance() -> dict[str, Any]:
    """Extract Git commit SHA, active branch, dirty state, and uncommitted diff hash."""
    git_data: dict[str, Any] = {
        "commit_sha": None,
        "branch": None,
        "is_dirty": False,
        "uncommitted_diff_sha256": None,
    }
    try:
        # Commit SHA
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                shell=False,
                text=True,
            )
            .strip()
        )
        git_data["commit_sha"] = sha

        # Branch name
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                shell=False,
                text=True,
            )
            .strip()
        )
        git_data["branch"] = branch

        # Dirty status
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                shell=False,
                text=True,
            )
            .strip()
        )
        if status:
            git_data["is_dirty"] = True
            diff = subprocess.check_output(
                ["git", "diff", "HEAD"],
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            git_data["uncommitted_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        else:
            git_data["is_dirty"] = False
            git_data["uncommitted_diff_sha256"] = None
    except Exception:
        # Fallback if git binary or repo metadata is not available
        pass

    return git_data


class BenchmarkManifestBuilder:
    """
    Constructs and serializes execution manifest (manifest.json).
    """

    def __init__(
        self,
        run_id: str,
        benchmark_name: str,
        argv: list[str] | None = None,
    ) -> None:
        self.run_id = run_id
        self.benchmark_name = benchmark_name
        self.argv = argv or sys.argv
        self.start_time = datetime.now(timezone.utc)
        self.end_time: datetime | None = None

        self.git_provenance = get_git_provenance()
        self.host_environment = get_host_environment()

        self.outcome: dict[str, Any] = {
            "status": "IN_PROGRESS",
            "exit_code": 0,
            "error_message": None,
            "total_matrix_cells": 0,
            "passed_cells": 0,
            "failed_oom_cells": 0,
            "failed_runtime_cells": 0,
        }

    def record_cell_result(self, status: str) -> None:
        """Update matrix cell completion counters."""
        self.outcome["total_matrix_cells"] += 1
        if status == "PASSED":
            self.outcome["passed_cells"] += 1
        elif status in ("FAILED_OOM", "OOM"):
            self.outcome["failed_oom_cells"] += 1
        else:
            self.outcome["failed_runtime_cells"] += 1

    def record_outcome(
        self,
        status: str = "COMPLETED",
        exit_code: int = 0,
        error_message: str | None = None,
        total_matrix_cells: int | None = None,
        passed_cells: int | None = None,
        failed_oom_cells: int | None = None,
        failed_runtime_cells: int | None = None,
    ) -> None:
        """Finalize outcome state."""
        self.end_time = datetime.now(timezone.utc)
        self.outcome["status"] = status
        self.outcome["exit_code"] = exit_code
        self.outcome["error_message"] = error_message
        if total_matrix_cells is not None:
            self.outcome["total_matrix_cells"] = total_matrix_cells
        if passed_cells is not None:
            self.outcome["passed_cells"] = passed_cells
        if failed_oom_cells is not None:
            self.outcome["failed_oom_cells"] = failed_oom_cells
        if failed_runtime_cells is not None:
            self.outcome["failed_runtime_cells"] = failed_runtime_cells

    def to_dict(self) -> dict[str, Any]:
        """Generate manifest dictionary adhering to Appendix B schema."""
        end_utc = self.end_time or datetime.now(timezone.utc)
        duration = max(0.0, (end_utc - self.start_time).total_seconds())

        return {
            "$schema": MANIFEST_SCHEMA_URL,
            "run_id": self.run_id,
            "benchmark_name": self.benchmark_name,
            "timestamp_start_utc": self.start_time.isoformat(),
            "timestamp_end_utc": end_utc.isoformat(),
            "duration_seconds": round(duration, 6),
            "argv": self.argv,
            "git_provenance": self.git_provenance,
            "host_environment": self.host_environment,
            "execution_outcome": self.outcome,
        }

    def save(self, path: Path | str) -> Path:
        """Atomically persist manifest.json to disk."""
        target_path = Path(path)
        manifest_file = (
            target_path if target_path.name.endswith(".json") else target_path / "manifest.json"
        )
        manifest_file.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        tmp_file = manifest_file.parent / f".tmp_{manifest_file.name}_{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, manifest_file)
        return manifest_file


def create_manifest(
    run_id: str,
    benchmark_name: str,
    output_dir: Path | str,
    status: str = "COMPLETED",
    exit_code: int = 0,
    error_message: str | None = None,
    total_cells: int = 0,
    passed_cells: int = 0,
    failed_oom_cells: int = 0,
    failed_runtime_cells: int = 0,
) -> dict[str, Any]:
    """Helper to construct and immediately write a manifest.json file."""
    builder = BenchmarkManifestBuilder(run_id=run_id, benchmark_name=benchmark_name)
    builder.record_outcome(
        status=status,
        exit_code=exit_code,
        error_message=error_message,
        total_matrix_cells=total_cells,
        passed_cells=passed_cells,
        failed_oom_cells=failed_oom_cells,
        failed_runtime_cells=failed_runtime_cells,
    )
    builder.save(output_dir)
    return builder.to_dict()
