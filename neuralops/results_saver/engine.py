#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Dual-Commit Persistence Engine

Coordinates simultaneous atomic persistence between:
1. Hermetic per-run isolated filesystem directory (summary.json, summary.csv, benchmark_report.md)
2. Master centralized SQLite ledger (benchmark_runs and benchmark_metrics tables)
Features retry with exponential backoff and fail-safe local pending commit backups.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from neuralops.benchmarks.core.paths import RunDirectoryContext, StoragePathManager
from neuralops.results_saver.storage.sqlite_backend import SQLiteMasterLedger
from neuralops.results_saver.writers.parquet_writer import write_metrics_parquet
from neuralops.results_saver.writers.rolling_aggregator import RollingSuiteAggregator
from neuralops.results_saver.writers.summary_generator import BenchmarkSummaryGenerator

logger = logging.getLogger(__name__)


class DualCommitPersistenceEngine:
    """Manages transactional dual-commit persistence for benchmark runs."""

    def __init__(
        self,
        ledger_path: Path | str = "results/master_ledger.sqlite",
        max_retries: int = 5,
        base_backoff_seconds: float = 0.1,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds

    def finalize_run(
        self,
        context: RunDirectoryContext,
        manifest: Dict[str, Any],
        summary: Optional[Dict[str, Any]] = None,
        iterations: Optional[List[Dict[str, Any]]] = None,
        duration_seconds: float = 0.0,
        enable_parquet: bool = True,
    ) -> bool:
        """
        Finalize a benchmark run by:
        1. Generating all run-local artifacts (summary.json, summary.csv, benchmark_report.md, metrics.parquet)
        2. Updating 'latest' symlink
        3. Appending to rolling suite summary.csv
        4. Transacting dual-commit into results/master_ledger.sqlite with retry & emergency backup
        """
        iterations = iterations or []

        # Step 1: Compute aggregates if not provided
        if summary is None or "aggregates" not in summary:
            aggregates = BenchmarkSummaryGenerator.compute_aggregates(iterations)
        else:
            aggregates = summary.get("aggregates", {})

        status = summary.get("status", "PASSED") if summary else ("PASSED" if aggregates.get("failed_runtime_iterations", 0) == 0 else "FAILED")

        # Step 2: Write run-local summary artifacts
        BenchmarkSummaryGenerator.generate_summary_json(
            run_id=context.run_id,
            benchmark_name=context.benchmark_name,
            aggregates=aggregates,
            manifest=manifest,
            iterations=iterations,
            target_path=context.summary_json,
        )

        if iterations:
            BenchmarkSummaryGenerator.generate_summary_csv(
                iterations=iterations,
                target_path=context.summary_csv,
                extra_fields={"run_id": context.run_id, "benchmark_name": context.benchmark_name},
            )

        BenchmarkSummaryGenerator.generate_markdown_report(
            run_id=context.run_id,
            benchmark_name=context.benchmark_name,
            aggregates=aggregates,
            manifest=manifest,
            iterations=iterations,
            target_path=context.report_md,
        )

        # Step 3: Optional Columnar Parquet export
        if enable_parquet and iterations:
            parquet_path = context.run_dir / "metrics.parquet"
            write_metrics_parquet(iterations, parquet_path)

        # Step 4: Update latest symlink
        StoragePathManager.update_latest_symlink(context)

        # Step 5: Append to rolling suite CSV
        suite_dir = context.run_dir.parent.parent
        RollingSuiteAggregator.append_run(
            suite_dir=suite_dir,
            run_id=context.run_id,
            benchmark_name=context.benchmark_name,
            aggregates=aggregates,
            manifest=manifest,
            duration_seconds=duration_seconds,
            status=status,
        )

        # Step 6: Commit to SQLite with exponential backoff & failover backup
        return self._commit_to_sqlite_with_failover(
            context=context,
            manifest=manifest,
            aggregates=aggregates,
            iterations=iterations,
            duration_seconds=duration_seconds,
            status=status,
        )

    def _commit_to_sqlite_with_failover(
        self,
        context: RunDirectoryContext,
        manifest: Dict[str, Any],
        aggregates: Dict[str, Any],
        iterations: List[Dict[str, Any]],
        duration_seconds: float,
        status: str,
    ) -> bool:
        """Attempt SQLite transaction with exponential backoff; dump emergency json on persistent failure."""
        git_sha = (manifest.get("git_provenance") or {}).get("commit_sha", "unknown")
        branch = (manifest.get("git_provenance") or {}).get("branch")
        is_dirty = 1 if (manifest.get("git_provenance") or {}).get("is_dirty") else 0
        device_name = manifest.get("device_name") or "unknown"
        hostname = manifest.get("hostname") or "unknown"
        tag = manifest.get("tag")

        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with SQLiteMasterLedger(db_path=str(self.ledger_path)) as ledger:
                    ledger.record_benchmark_run(
                        run_id=context.run_id,
                        benchmark_name=context.benchmark_name,
                        commit_sha=git_sha,
                        hostname=hostname,
                        device_name=device_name,
                        run_dir=str(context.run_dir),
                        status=status,
                        suite=context.benchmark_name,
                        timestamp_start_utc=manifest.get("timestamp_start_utc"),
                        timestamp_end_utc=manifest.get("timestamp_end_utc"),
                        duration_seconds=duration_seconds,
                        branch=branch,
                        is_dirty=is_dirty,
                        tag=tag,
                        total_cells=aggregates.get("total_iterations", 0),
                        passed_cells=aggregates.get("passed_iterations", 0),
                        failed_cells=aggregates.get("failed_runtime_iterations", 0) + aggregates.get("failed_oom_iterations", 0),
                    )

                    if iterations:
                        metrics_to_commit = []
                        for i, it in enumerate(iterations):
                            metrics_to_commit.append({
                                "run_id": context.run_id,
                                "step": it.get("step", i + 1),
                                "mechanism_or_kernel": it.get("mechanism_or_kernel", it.get("module", "default")),
                                "batch_size": it.get("batch_size"),
                                "seq_len": it.get("seq_len"),
                                "dtype": it.get("dtype"),
                                "latency_p50_ms": it.get("latency_ms") or it.get("latency_p50_ms"),
                                "latency_p90_ms": it.get("latency_p90_ms"),
                                "latency_p99_ms": it.get("latency_p99_ms"),
                                "tflops": it.get("tflops"),
                                "peak_vram_mb": it.get("vram_mb") or it.get("peak_vram_mb"),
                                "extra_metadata": it.get("extra_metadata"),
                            })
                        ledger.record_benchmark_metrics(metrics_to_commit)

                return True
            except Exception as exc:
                last_exception = exc
                sleep_sec = self.base_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "SQLite transaction attempt %d failed: %s. Retrying in %.3fs...",
                    attempt,
                    exc,
                    sleep_sec,
                )
                time.sleep(sleep_sec)

        # If all retries failed, write emergency local pending commit file
        emergency_file = context.run_dir / "sqlite_pending_commit.json"
        emergency_payload = {
            "run_id": context.run_id,
            "benchmark_name": context.benchmark_name,
            "manifest": manifest,
            "aggregates": aggregates,
            "iterations": iterations,
            "error": str(last_exception),
        }
        emergency_file.write_text(json.dumps(emergency_payload, indent=2, default=str), encoding="utf-8")
        logger.error(
            "SQLite transaction failed after %d attempts; wrote emergency backup to %s",
            self.max_retries,
            emergency_file,
        )
        return False
