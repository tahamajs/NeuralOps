#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""SQLite Backend for Master Ledger Storage and Cryptographic Provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class SQLiteMasterLedger:
    """Unified interface for SQLite master ledger with SHA-256 provenance."""

    def __init__(self, db_path: str = "results/master_ledger.sqlite") -> None:
        self.db_path = Path(db_path)
        if not str(self.db_path) == ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def __enter__(self) -> SQLiteMasterLedger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        """Initialize master ledger schema and relational indexes."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hardware_telemetry (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_name TEXT,
                compute_capability TEXT,
                vram_total_gb REAL,
                driver_version TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS master_benchmark_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                architecture_name TEXT,
                compression_method TEXT,
                batch_size INTEGER,
                seq_len INTEGER,
                latency_ms REAL,
                throughput_tokens_per_sec REAL,
                peak_vram_mb REAL,
                accuracy_pct REAL,
                sha256_provenance TEXT,
                FOREIGN KEY (run_id) REFERENCES hardware_telemetry (run_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                benchmark_name TEXT NOT NULL,
                suite TEXT DEFAULT '',
                timestamp_utc TEXT,
                timestamp_start_utc TEXT,
                timestamp_end_utc TEXT,
                duration_s REAL,
                duration_seconds REAL,
                commit_sha TEXT NOT NULL,
                branch TEXT,
                is_dirty INTEGER DEFAULT 0,
                hostname TEXT NOT NULL,
                device_name TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                status TEXT NOT NULL,
                tag TEXT,
                total_cells INTEGER DEFAULT 0,
                passed_cells INTEGER DEFAULT 0,
                failed_cells INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER DEFAULT 0,
                mechanism_or_kernel TEXT NOT NULL,
                batch_size INTEGER,
                seq_len INTEGER,
                dtype TEXT,
                latency_p50_ms REAL,
                latency_p90_ms REAL,
                latency_p99_ms REAL,
                tflops REAL,
                peak_vram_mb REAL,
                extra_metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_runs_suite ON benchmark_runs(suite)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_runs_timestamp ON benchmark_runs(timestamp_start_utc)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_metrics_run ON benchmark_metrics(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_metrics_mechanism ON benchmark_metrics(mechanism_or_kernel)")
        self.conn.commit()

    def record_run(
        self,
        device_name: str,
        compute_capability: str,
        vram_total_gb: float,
        driver_version: str,
    ) -> int:
        """Insert hardware run record and return run_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO hardware_telemetry (device_name, compute_capability, vram_total_gb, driver_version) VALUES (?, ?, ?, ?)",
            (device_name, compute_capability, vram_total_gb, driver_version),
        )
        self.conn.commit()
        return cursor.lastrowid

    def record_result(
        self,
        run_id: int,
        architecture_name: str,
        compression_method: str,
        batch_size: int,
        seq_len: int,
        latency_ms: float,
        throughput_tokens_per_sec: float,
        peak_vram_mb: float,
        accuracy_pct: float,
    ) -> Tuple[int, str]:
        """Insert benchmark result with deterministic SHA-256 hash."""
        raw_payload = f"{architecture_name}:{compression_method}:{batch_size}:{seq_len}:{latency_ms:.4f}:{accuracy_pct:.2f}"
        sha_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO master_benchmark_results (
                run_id, architecture_name, compression_method, batch_size, seq_len,
                latency_ms, throughput_tokens_per_sec, peak_vram_mb, accuracy_pct, sha256_provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                architecture_name,
                compression_method,
                batch_size,
                seq_len,
                latency_ms,
                throughput_tokens_per_sec,
                peak_vram_mb,
                accuracy_pct,
                sha_hash,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid, sha_hash

    def record_benchmark_run(
        self,
        run_id: str,
        benchmark_name: str,
        commit_sha: str,
        hostname: str,
        device_name: str,
        run_dir: str,
        status: str,
        suite: str = "",
        timestamp_start_utc: Optional[str] = None,
        timestamp_end_utc: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        branch: Optional[str] = None,
        is_dirty: int = 0,
        tag: Optional[str] = None,
        total_cells: int = 0,
        passed_cells: int = 0,
        failed_cells: int = 0,
    ) -> None:
        """Insert or replace benchmark run metadata into master ledger."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO benchmark_runs (
                run_id, benchmark_name, suite, timestamp_utc, timestamp_start_utc,
                timestamp_end_utc, duration_s, duration_seconds, commit_sha,
                branch, is_dirty, hostname, device_name, run_dir, status,
                tag, total_cells, passed_cells, failed_cells
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                benchmark_name,
                suite or benchmark_name,
                timestamp_start_utc,
                timestamp_start_utc,
                timestamp_end_utc,
                duration_seconds,
                duration_seconds,
                commit_sha,
                branch,
                is_dirty,
                hostname,
                device_name,
                run_dir,
                status,
                tag,
                total_cells,
                passed_cells,
                failed_cells,
            ),
        )
        self.conn.commit()

    def record_benchmark_metrics(
        self,
        metrics: List[Dict[str, Any]],
    ) -> None:
        """Bulk insert benchmark metric iterations into master ledger."""
        cursor = self.conn.cursor()
        rows = []
        for m in metrics:
            extra = m.get("extra_metadata")
            if isinstance(extra, dict):
                extra_str = json.dumps(extra)
            else:
                extra_str = str(extra) if extra is not None else None
            rows.append((
                m["run_id"],
                m.get("step", 0),
                m["mechanism_or_kernel"],
                m.get("batch_size"),
                m.get("seq_len"),
                m.get("dtype"),
                m.get("latency_p50_ms"),
                m.get("latency_p90_ms"),
                m.get("latency_p99_ms"),
                m.get("tflops"),
                m.get("peak_vram_mb"),
                extra_str,
            ))
        cursor.executemany(
            """
            INSERT INTO benchmark_metrics (
                run_id, step, mechanism_or_kernel, batch_size, seq_len, dtype,
                latency_p50_ms, latency_p90_ms, latency_p99_ms, tflops,
                peak_vram_mb, extra_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def get_benchmark_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a single benchmark run by run_id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            return None
        col_names = [d[0] for d in cursor.description]
        return dict(zip(col_names, row))

    def get_benchmark_metrics(self, run_id: str) -> List[Dict[str, Any]]:
        """Fetch step-level iteration metrics for a run."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM benchmark_metrics WHERE run_id = ? ORDER BY step", (run_id,))
        rows = cursor.fetchall()
        if not rows:
            return []
        col_names = [d[0] for d in cursor.description]
        return [dict(zip(col_names, r)) for r in rows]

    def list_benchmark_runs(self, suite: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent benchmark runs, optionally filtered by suite."""
        cursor = self.conn.cursor()
        if suite:
            cursor.execute(
                "SELECT * FROM benchmark_runs WHERE suite = ? OR benchmark_name = ? ORDER BY timestamp_start_utc DESC LIMIT ?",
                (suite, suite, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM benchmark_runs ORDER BY timestamp_start_utc DESC LIMIT ?",
                (limit,),
            )
        rows = cursor.fetchall()
        if not rows:
            return []
        col_names = [d[0] for d in cursor.description]
        return [dict(zip(col_names, r)) for r in rows]

    def commit(self) -> None:
        """Commit active transaction."""
        self.conn.commit()

    def close(self) -> None:
        """Close connection to SQLite database."""
        self.conn.close()


def init_master_ledger(db_path: str = "results/master_ledger.sqlite") -> SQLiteMasterLedger:
    """Convenience initializer ensuring canonical schema exists on database."""
    return SQLiteMasterLedger(db_path=db_path)
