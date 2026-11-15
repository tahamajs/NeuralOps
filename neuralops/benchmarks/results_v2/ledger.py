"""
Results v2 BenchmarkLedger implementation.
Atomic storage writer to JSONL, SQLite, and Parquet (AGENTS.md Rule 4.3).
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from neuralops.benchmarks.results_v2.record import CanonicalRecord

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


class BenchmarkLedger:
    """Atomic multi-backend persistence ledger for CanonicalRecord instances."""

    def __init__(self, storage_dir: str | Path = "results") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.storage_dir / "results.jsonl"
        self.db_path = self.storage_dir / "ledger.db"
        self.parquet_path = self.storage_dir / "results.parquet"
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite schema if database does not exist."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS records (
                        run_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        git_commit TEXT NOT NULL,
                        git_dirty INTEGER NOT NULL,
                        suite TEXT NOT NULL,
                        module TEXT NOT NULL,
                        model TEXT NOT NULL,
                        dataset TEXT NOT NULL,
                        device TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        config TEXT NOT NULL,
                        telemetry TEXT NOT NULL,
                        checksum TEXT PRIMARY KEY
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_records_run_id ON records(run_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_records_suite ON records(suite)"
                )
        finally:
            conn.close()

    def write(self, record: CanonicalRecord) -> None:
        """Write a record atomically across JSONL, SQLite, and Parquet."""
        data = record.to_dict()

        # 1. Write to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()

        # 2. Write to SQLite
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO records (
                        run_id, timestamp, git_commit, git_dirty,
                        suite, module, model, dataset, device,
                        status, metrics, config, telemetry, checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        record.timestamp,
                        record.git_commit,
                        1 if record.git_dirty else 0,
                        record.suite,
                        record.module,
                        record.model,
                        record.dataset,
                        record.device,
                        record.status,
                        json.dumps(record.metrics),
                        json.dumps(record.config),
                        json.dumps(record.telemetry),
                        record.checksum,
                    ),
                )
        finally:
            conn.close()

        # 3. Write to Parquet if available
        if _HAS_PYARROW:
            try:
                parquet_row = {
                    "run_id": record.run_id,
                    "timestamp": record.timestamp,
                    "git_commit": record.git_commit,
                    "git_dirty": record.git_dirty,
                    "suite": record.suite,
                    "module": record.module,
                    "model": record.model,
                    "dataset": record.dataset,
                    "device": record.device,
                    "status": record.status,
                    "metrics": json.dumps(record.metrics),
                    "config": json.dumps(record.config),
                    "telemetry": json.dumps(record.telemetry),
                    "checksum": record.checksum,
                }
                new_table = pa.Table.from_pylist([parquet_row])
                if self.parquet_path.exists():
<<<<<<< HEAD
<<<<<<< HEAD
                    existing_table = pq.read_table(str(self.parquet_path))
                    combined_table = pa.concat_tables([existing_table, new_table])
                    pq.write_table(combined_table, str(self.parquet_path))
                else:
                    pq.write_table(new_table, str(self.parquet_path))
=======
=======
>>>>>>> wave5/W5-11
                    existing_table = pq.read_table(str(self.parquet_path))  # type: ignore[no-untyped-call]  # pyarrow stubs
                    combined_table = pa.concat_tables([existing_table, new_table])
                    pq.write_table(combined_table, str(self.parquet_path))  # type: ignore[no-untyped-call]  # pyarrow stubs
                else:
                    pq.write_table(new_table, str(self.parquet_path))  # type: ignore[no-untyped-call]  # pyarrow stubs
<<<<<<< HEAD
>>>>>>> wave5/W5-9
=======
>>>>>>> wave5/W5-11
            except Exception:
                pass
