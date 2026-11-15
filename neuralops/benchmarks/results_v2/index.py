"""
Query index over benchmark results storage (Results v2).
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from neuralops.benchmarks.results_v2.record import CanonicalRecord


class ResultsIndex:
    """Indexed search and querying across historical benchmark executions."""

    def __init__(self, storage_dir: str | Path = "results") -> None:
        self.storage_dir = Path(storage_dir)
        self.db_path = self.storage_dir / "ledger.db"
        self.jsonl_path = self.storage_dir / "results.jsonl"

    def find(
        self,
        run_id: str | None = None,
        suite: str | None = None,
        git_commit: str | None = None,
        status: str | None = None,
        device: str | None = None,
    ) -> list[CanonicalRecord]:
        """Find and return CanonicalRecord instances matching specified criteria."""
        if self.db_path.exists():
            return self._find_sqlite(
                run_id=run_id,
                suite=suite,
                git_commit=git_commit,
                status=status,
                device=device,
            )
        if self.jsonl_path.exists():
            return self._find_jsonl(
                run_id=run_id,
                suite=suite,
                git_commit=git_commit,
                status=status,
                device=device,
            )
        return []

    def _find_sqlite(
        self,
        run_id: str | None = None,
        suite: str | None = None,
        git_commit: str | None = None,
        status: str | None = None,
        device: str | None = None,
    ) -> list[CanonicalRecord]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            query = "SELECT * FROM records WHERE 1=1"
            params: list[Any] = []
            if run_id is not None:
                query += " AND run_id = ?"
                params.append(run_id)
            if suite is not None:
                query += " AND suite = ?"
                params.append(suite)
            if git_commit is not None:
                query += " AND git_commit = ?"
                params.append(git_commit)
            if status is not None:
                query += " AND status = ?"
                params.append(status)
            if device is not None:
                query += " AND device = ?"
                params.append(device)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            records: list[CanonicalRecord] = []
            for row in rows:
                row_dict = dict(row)
                records.append(CanonicalRecord.from_dict(row_dict))
            return records
        finally:
            conn.close()

    def _find_jsonl(
        self,
        run_id: str | None = None,
        suite: str | None = None,
        git_commit: str | None = None,
        status: str | None = None,
        device: str | None = None,
    ) -> list[CanonicalRecord]:
        records: list[CanonicalRecord] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if run_id is not None and data.get("run_id") != run_id:
                    continue
                if suite is not None and data.get("suite") != suite:
                    continue
                if git_commit is not None and data.get("git_commit") != git_commit:
                    continue
                if status is not None and data.get("status") != status:
                    continue
                if device is not None and data.get("device") != device:
                    continue
                records.append(CanonicalRecord.from_dict(data))
        return records
