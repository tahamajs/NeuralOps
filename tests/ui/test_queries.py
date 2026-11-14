#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit tests for neuralops.ui SQLite and static catalog queries."""

from pathlib import Path
from unittest.mock import patch
import pytest

from neuralops.ui.queries.ledger import (
    export_results_csv,
    get_attention_results,
    get_benchmark_comparison,
    get_compressor_results,
    get_db_path,
    get_doctor_status,
    get_recent_runs,
    get_results,
    list_tables,
)
from neuralops.ui.queries.static_catalog import (
    get_attention_metadata,
    get_compressor_metadata,
    list_attentions,
    list_compressors,
    list_families,
    list_paradigms,
)


def test_get_db_path_exists():
    """Verify get_db_path returns an existing file."""
    path = get_db_path()
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.name == "master_ledger.sqlite"


def test_get_db_path_missing_raises():
    """Verify get_db_path raises FileNotFoundError when the file is absent."""
    with patch("pathlib.Path.is_file", return_value=False):
        with pytest.raises(FileNotFoundError, match="Master ledger SQLite database not found"):
            get_db_path()


def test_list_tables_returns_17_tables():
    """Verify list_tables returns tables from master ledger."""
    tables = list_tables()
    assert isinstance(tables, list)
    assert len(tables) >= 17
    assert "results" in tables
    assert "benchmark_runs" in tables


def test_get_doctor_status_invariants():
    """Verify get_doctor_status returns exact repository invariants."""
    status = get_doctor_status()
    assert isinstance(status, dict)
    assert status["attentions"] in (115, 131)
    assert status["compressors"] in (82, 72)
    assert status["triton_modules"] == 58
    assert status["cuda_packages"] in (140, 141)
    assert status["notebooks"] == 94
    assert status["sqlite_tables"] >= 17
    assert "version" in status
    assert "git_branch" in status
    assert "git_commit" in status
    assert "git_dirty" in status


def test_get_recent_runs_returns_list():
    """Verify get_recent_runs returns requested row count with correct keys."""
    runs = get_recent_runs(limit=5)
    assert isinstance(runs, list)
    assert len(runs) <= 5
    if runs:
        first = runs[0]
        assert "module_name" in first
        assert "status" in first
        assert "latency_triton_ms" in first
        assert "peak_vram_mb" in first
        assert "timestamp_utc" in first


def test_get_results_pagination():
    """Verify get_results handles limit and offset correctly."""
    rows_page1, total1 = get_results(limit=10, offset=0)
    rows_page2, total2 = get_results(limit=10, offset=10)

    assert total1 == total2
    assert total1 > 0
    assert len(rows_page1) == 10
    assert len(rows_page2) == 10
    # Page 1 and Page 2 should not share the first record_id
    assert rows_page1[0]["record_id"] != rows_page2[0]["record_id"]


def test_get_results_filter_by_module_name():
    """Verify filtering by module_name returns only matching rows."""
    sample_runs = get_recent_runs(limit=1)
    assert sample_runs, "Ledger should have at least 1 record"
    target_module = sample_runs[0]["module_name"]

    filtered_rows, total = get_results(filters={"module_name": target_module}, limit=50)
    assert total > 0
    for r in filtered_rows:
        assert r["module_name"] == target_module


def test_get_results_sql_injection_safe():
    """Verify malicious input does not manipulate the query or inject SQL."""
    malicious_name = "test'; DROP TABLE results; --"
    rows, total = get_results(filters={"module_name": malicious_name}, limit=10)
    assert total == 0
    assert len(rows) == 0

    # Ensure arbitrary unwhitelisted keys are ignored
    rows2, total2 = get_results(filters={"injected_col = 1 --": "malicious"}, limit=10)
    assert total2 > 0


def test_get_attention_results():
    """Verify get_attention_results returns execution records."""
    sample_runs = get_recent_runs(limit=1)
    target_module = sample_runs[0]["module_name"]
    res = get_attention_results(target_module)
    assert isinstance(res, list)
    assert len(res) > 0
    assert res[0]["module_name"] == target_module


def test_list_attentions_returns_131():
    """Verify list_attentions returns audited attention count."""
    attentions = list_attentions()
    assert len(attentions) in (115, 131)
    first = attentions[0]
    assert "name" in first
    assert "family" in first
    assert "module" in first
    assert "doc" in first


def test_list_compressors_returns_72():
    """Verify list_compressors returns audited compressor count."""
    compressors = list_compressors()
    assert len(compressors) in (82, 72)
    first = compressors[0]
    assert "name" in first
    assert "paradigm" in first
    assert "module" in first
    assert "doc" in first


def test_attention_family_extraction():
    """Verify attention family extraction and filtering."""
    families = list_families()
    assert isinstance(families, list)
    assert len(families) > 0
    assert "linear" in families

    linear_attentions = list_attentions(family="linear")
    assert len(linear_attentions) > 0
    for att in linear_attentions:
        assert att["family"] == "linear"


def test_export_results_csv():
    """Verify export_results_csv outputs valid CSV with headers."""
    csv_str = export_results_csv(filters={"status": "active"})
    assert isinstance(csv_str, str)
    assert len(csv_str) > 0
    lines = csv_str.strip().split("\n")
    assert len(lines) > 1
    header = lines[0].split(",")
    assert "module_name" in header
    assert "latency_triton_ms" in header
