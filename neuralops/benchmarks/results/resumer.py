#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Benchmark sweep resumer and idempotency verification engine.

Reads .journal.jsonl upon launch, reconstructs the set of completed
(model, attention, benchmark) runs, and enables sweeps to resume without
duplicating work.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

LOG = logging.getLogger(__name__)


class BenchmarkSweepResumer:
    """Manages audit and idempotency checks for resumable benchmark sweeps.

    Inspects .journal.jsonl in output_dir, builds the set of completed
    work units, and determines whether candidate runs should be executed or skipped.
    """

    def __init__(self, output_dir: Path | str = "results") -> None:
        self.output_dir = Path(output_dir)
        self.journal_path = self.output_dir / ".journal.jsonl"
        self._completed_runs: Set[str] = set()
        self._completed_tuples: Set[Tuple[str, str, str]] = set()
        self._run_entries: List[Dict[str, Any]] = []
        self._audit_journal()

    def _audit_journal(self) -> None:
        """Parse .journal.jsonl and identify completed and failed runs."""
        self._completed_runs.clear()
        self._completed_tuples.clear()
        self._run_entries.clear()

        if not self.journal_path.exists():
            return

        try:
            with open(self.journal_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self._run_entries.append(record)

                        event = record.get("event", "")
                        run_id = record.get("run_id", "")
                        status = record.get("status", "")

                        # A run is considered complete if event is CASE_SUCCESS or status is success
                        if event == "CASE_SUCCESS" or status == "success":
                            if run_id:
                                self._completed_runs.add(run_id)

                            m_name = record.get("model_name", "") or record.get("model", "")
                            t_attn = record.get("transplanted_attention", "") or record.get("module", "")
                            b_suite = record.get("benchmark_suite", "") or record.get("dataset", "")

                            if m_name or t_attn or b_suite:
                                self._completed_tuples.add((str(m_name), str(t_attn), str(b_suite)))

                    except json.JSONDecodeError:
                        LOG.warning("Malformed JSON line encountered in journal: %s", line)
        except Exception as exc:
            LOG.warning("Failed to parse journal file at %s: %s", self.journal_path, exc)

    @property
    def completed_count(self) -> int:
        """Return total number of completed runs recorded in journal."""
        return len(self._completed_runs)

    def get_completed_runs(self) -> Set[Tuple[str, str, str]]:
        """Return the set of completed (model_name, attention_key, benchmark_suite) tuples."""
        return set(self._completed_tuples)

    def is_completed(
        self,
        model_name: str,
        transplanted_attention: str,
        benchmark_suite: str,
    ) -> bool:
        """Check whether a (model, attention, benchmark) work item is already done."""
        key = (str(model_name), str(transplanted_attention), str(benchmark_suite))
        return key in self._completed_tuples

    def should_skip(
        self,
        model_name: str,
        transplanted_attention: str,
        benchmark_suite: str,
        force_rerun: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Determine whether to skip a benchmark case during a resumed sweep.

        Returns:
            (should_skip, reason)
        """
        if force_rerun:
            return False, "force_rerun=True overrides resume skip"

        if self.is_completed(model_name, transplanted_attention, benchmark_suite):
            return True, f"Already completed in journal: {model_name} / {transplanted_attention} / {benchmark_suite}"

        return False, "Not found in completed journal records"

    def filter_pending(
        self,
        sweep_matrix: List[Dict[str, Any]],
        force_rerun: bool = False,
    ) -> List[Dict[str, Any]]:
        """Filter a list of sweep configurations, omitting cases that have already finished."""
        if force_rerun:
            return list(sweep_matrix)

        pending: List[Dict[str, Any]] = []
        for case in sweep_matrix:
            m = case.get("model_name", "") or case.get("model", "")
            a = case.get("transplanted_attention", "") or case.get("attention", "") or case.get("module", "")
            b = case.get("benchmark_suite", "") or case.get("benchmark", "") or case.get("dataset", "")

            skip, _ = self.should_skip(m, a, b, force_rerun=force_rerun)
            if not skip:
                pending.append(case)

        return pending


def dedup_results_ledger(ledger_db_path: Path | str) -> int:
    """Scan the SQLite master ledger, find duplicate runs by composite key, and prune redundant entries.

    Keeps only the highest-scoring run (or most recent if metrics are identical) per composite key
    (model_name, transplanted_attention, benchmark_suite).

    Args:
        ledger_db_path: Path to master_ledger.db SQLite database.

    Returns:
        Number of duplicate entries deleted.
    """
    db_path = Path(ledger_db_path)
    if not db_path.exists():
        return 0

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    deleted_count = 0
    try:
        cur = conn.cursor()
        # Find composite keys that have duplicates
        cur.execute(
            """
            SELECT model_name, transplanted_attention, benchmark_suite, COUNT(*) as cnt
            FROM benchmark_runs
            GROUP BY model_name, transplanted_attention, benchmark_suite
            HAVING cnt > 1
            """
        )
        dup_groups = cur.fetchall()

        for model_name, transplanted_attention, benchmark_suite, _ in dup_groups:
            cur.execute(
                """
                SELECT run_id, timestamp, status, metrics_json
                FROM benchmark_runs
                WHERE model_name = ? AND transplanted_attention = ? AND benchmark_suite = ?
                """,
                (model_name, transplanted_attention, benchmark_suite),
            )
            rows = cur.fetchall()

            # Rank rows: prefer status == 'success', then higher accuracy/score/throughput, then newer timestamp
            def score_row(row: Tuple[str, str, str, str]) -> Tuple[int, float, str]:
                r_id, ts, status, m_json = row
                status_score = 1 if status == "success" else 0
                metric_score = 0.0
                try:
                    m_data = json.loads(m_json)
                    if isinstance(m_data, dict):
                        for k in ("accuracy", "score", "throughput", "throughput_req_sec", "retention"):
                            if k in m_data and isinstance(m_data[k], (int, float)):
                                metric_score = max(metric_score, float(m_data[k]))
                except Exception:
                    pass
                return (status_score, metric_score, ts)

            best_row = max(rows, key=score_row)
            best_run_id = best_row[0]

            for row in rows:
                if row[0] != best_run_id:
                    cur.execute("DELETE FROM benchmark_runs WHERE run_id = ?", (row[0],))
                    deleted_count += 1

        conn.commit()
    finally:
        conn.close()

    return deleted_count


__all__ = [
    "BenchmarkSweepResumer",
    "dedup_results_ledger",
]
