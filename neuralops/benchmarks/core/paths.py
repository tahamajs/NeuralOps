#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hermetic per-run storage path manager and atomic symlink coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import os
from pathlib import Path
import re
from typing import Optional
import uuid


@dataclass(frozen=True)
class RunDirectoryContext:
    """Immutable context encapsulating isolated filesystem locations for a run."""

    run_id: str
    benchmark_name: str
    run_dir: Path
    artifacts_dir: Path
    tensorboard_dir: Path
    metrics_jsonl: Path
    summary_json: Path
    summary_csv: Path
    manifest_json: Path
    config_resolved_yaml: Path
    log_file: Path
    report_md: Path


class StoragePathManager:
    """Manages the creation and isolation of per-run storage directories."""

    @staticmethod
    def generate_run_id() -> str:
        """Generate a deterministic, sortable, collision-free run identifier."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_suffix}"

    @classmethod
    def resolve_run_dir(
        cls,
        benchmark_name: str,
        run_id: Optional[str] = None,
        output_root: Optional[Path | str] = None,
    ) -> RunDirectoryContext:
        """
        Create and return an isolated per-run directory context:
        results/benchmarks/<benchmark_name>/runs/<run_id>/
        """
        root = Path(output_root) if output_root is not None else Path("results/benchmarks")
        clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", benchmark_name.lower().strip().replace(" ", "_"))
        active_run_id = run_id or cls.generate_run_id()

        suite_dir = root / clean_name
        run_dir = suite_dir / "runs" / active_run_id
        artifacts_dir = run_dir / "artifacts"
        tensorboard_dir = run_dir / "tensorboard"

        # Create isolated directory tree atomically
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        return RunDirectoryContext(
            run_id=active_run_id,
            benchmark_name=clean_name,
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            tensorboard_dir=tensorboard_dir,
            metrics_jsonl=run_dir / "metrics.jsonl",
            summary_json=run_dir / "summary.json",
            summary_csv=run_dir / "summary.csv",
            manifest_json=run_dir / "manifest.json",
            config_resolved_yaml=run_dir / "config_resolved.yaml",
            log_file=run_dir / "stdout_stderr.log",
            report_md=run_dir / "benchmark_report.md",
        )

    @staticmethod
    def update_latest_symlink(context: RunDirectoryContext) -> None:
        """
        Atomically update 'latest' symlink in suite root pointing to the current run.
        """
        suite_dir = context.run_dir.parent.parent
        latest_link = suite_dir / "latest"
        target_relative = Path("runs") / context.run_id

        temp_link = suite_dir / f".tmp_latest_{context.run_id}"
        try:
            if temp_link.is_symlink() or temp_link.exists():
                temp_link.unlink()

            os.symlink(target_relative, temp_link)
            os.replace(temp_link, latest_link)
        except OSError:
            # Cross-platform fallback for environments restricting symlinks
            latest_txt = suite_dir / "latest.txt"
            with open(latest_txt, "w", encoding="utf-8") as f:
                f.write(str(target_relative) + "\n")
        finally:
            if temp_link.is_symlink() or temp_link.exists():
                try:
                    temp_link.unlink()
                except OSError:
                    pass

    @staticmethod
    def get_latest_run_dir(
        benchmark_name: str,
        output_root: Optional[Path | str] = None,
    ) -> Optional[Path]:
        """Resolve the latest run directory for a benchmark suite if it exists."""
        root = Path(output_root) if output_root is not None else Path("results/benchmarks")
        clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", benchmark_name.lower().strip().replace(" ", "_"))
        suite_dir = root / clean_name
        latest_link = suite_dir / "latest"
        if latest_link.is_symlink():
            target = (suite_dir / os.readlink(latest_link)).resolve()
            if target.exists():
                return target
        latest_txt = suite_dir / "latest.txt"
        if latest_txt.exists():
            rel_path = latest_txt.read_text(encoding="utf-8").strip()
            target = (suite_dir / rel_path).resolve()
            if target.exists():
                return target
        return None
