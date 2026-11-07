#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SQLite Maintenance & Retention Policy Engine

Provides operational utilities to:
1. Verify database integrity (PRAGMA integrity_check)
2. Recover disk space and rebuild indexes (VACUUM, REINDEX)
3. Enforce lifecycle retention policy across historical per-run metrics.
"""

from __future__ import annotations

import datetime
import gzip
import logging
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SQLiteMaintenanceEngine:
    """Manages SQLite database health and per-run metric retention schedules."""

    @staticmethod
    def check_integrity(db_path: Path | str = "results/master_ledger.sqlite") -> Dict[str, Any]:
        """Perform full integrity check on master SQLite ledger."""
        p = Path(db_path)
        if not p.exists():
            return {"status": "ERROR", "message": f"Database file not found: {p}"}

        conn = sqlite3.connect(str(p))
        try:
            cursor = conn.cursor()
            results = cursor.execute("PRAGMA integrity_check;").fetchall()
            status = "OK" if len(results) == 1 and results[0][0] == "ok" else "CORRUPTED"
            return {
                "status": status,
                "db_path": str(p),
                "details": [r[0] for r in results],
            }
        finally:
            conn.close()

    @staticmethod
    def vacuum_and_reindex(db_path: Path | str = "results/master_ledger.sqlite") -> Dict[str, Any]:
        """Vacuum database to reclaim disk space and rebuild all indexes."""
        p = Path(db_path)
        if not p.exists():
            return {"status": "ERROR", "message": f"Database file not found: {p}"}

        initial_size = p.stat().st_size
        conn = sqlite3.connect(str(p))
        try:
            cursor = conn.cursor()
            cursor.execute("VACUUM;")
            cursor.execute("REINDEX;")
            conn.commit()
        finally:
            conn.close()

        final_size = p.stat().st_size
        reclaimed_bytes = max(0, initial_size - final_size)

        return {
            "status": "SUCCESS",
            "initial_size_bytes": initial_size,
            "final_size_bytes": final_size,
            "reclaimed_bytes": reclaimed_bytes,
        }

    @staticmethod
    def enforce_retention_policy(
        benchmark_root: Path | str = "results/benchmarks",
        max_age_days: int = 30,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """
        Compress or remove high-frequency metrics.jsonl older than max_age_days
        while strictly preserving manifest.json, summary.json, and SQLite records forever.
        """
        root = Path(benchmark_root)
        if not root.exists():
            return {"processed_runs": 0, "compressed_files": 0, "unlinked_files": 0}

        now_ts = time.time()
        max_age_seconds = max_age_days * 86400
        compressed_count = 0
        unlinked_count = 0
        processed_runs: List[str] = []

        for runs_dir in root.glob("*/runs"):
            if not runs_dir.is_dir():
                continue
            for run_path in runs_dir.iterdir():
                if not run_path.is_dir():
                    continue

                mtime = run_path.stat().st_mtime
                age = now_ts - mtime
                if age > max_age_seconds:
                    processed_runs.append(str(run_path))
                    metrics_jsonl = run_path / "metrics.jsonl"
                    if metrics_jsonl.exists():
                        if compress:
                            gz_dest = run_path / "metrics.jsonl.gz"
                            with open(metrics_jsonl, "rb") as f_in, gzip.open(gz_dest, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                            metrics_jsonl.unlink()
                            compressed_count += 1
                        else:
                            metrics_jsonl.unlink()
                            unlinked_count += 1

        return {
            "cutoff_date_utc": (
                datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=max_age_days)
            ).isoformat(),
            "processed_runs": len(processed_runs),
            "compressed_files": compressed_count,
            "unlinked_files": unlinked_count,
        }
