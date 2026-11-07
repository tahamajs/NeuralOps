#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying single persistence owner for results ledger and configurable UI path (F-59).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import pytest

from neuralops.ui.queries.ledger import get_db_path, list_tables


def test_ui_reads_ledger_via_configured_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify UI resolves ledger path through NEURALOPS_LEDGER_PATH and reads custom database."""
    test_db = tmp_path / "custom_ledger.sqlite"
    conn = sqlite3.connect(str(test_db))
    conn.execute("CREATE TABLE test_runs (id INTEGER PRIMARY KEY, status TEXT)")
    conn.commit()
    conn.close()

    monkeypatch.setenv("NEURALOPS_LEDGER_PATH", str(test_db))

    resolved_path = get_db_path()
    assert resolved_path == test_db.resolve()

    tables = list_tables()
    assert "test_runs" in tables


def test_single_ledger_writer() -> None:
    """Verify results_saver is the declared single persistence owner and reporting has no write queries."""
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Reporting modules must not execute mutating SQL queries
    reporting_dir = repo_root / "neuralops" / "reporting"
    for py_file in reporting_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "INSERT INTO" not in content, f"Mutating query found in {py_file}"
        assert "UPDATE " not in content, f"Mutating query found in {py_file}"
        assert "DELETE FROM" not in content, f"Mutating query found in {py_file}"

    # 2. analysis/results_ledger.py and reporting docstrings must declare single ownership under results_saver
    analysis_ledger = repo_root / "neuralops" / "analysis" / "results_ledger.py"
    assert "results_saver" in analysis_ledger.read_text(encoding="utf-8")

    evidence_file = repo_root / "neuralops" / "reporting" / "evidence.py"
    assert "results_saver" in evidence_file.read_text(encoding="utf-8")
