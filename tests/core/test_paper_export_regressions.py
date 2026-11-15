#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for neuralops.core.paper_export."""

from __future__ import annotations

import csv
import io
import pytest

from neuralops.core import paper_export
from neuralops.core.results_ledger import ResultRecord


def _sample_records():
    return [
        ResultRecord(
            record_id="rec_001",
            module_name="flash_attention",
            category="attention",
            family="fast_kernels",
            metrics={"latency_pytorch_ms": 1.25, "latency_triton_ms": 0.85, "speedup_x": 1.47},
        ),
        ResultRecord(
            record_id="rec_002",
            module_name="gla",
            category="attention",
            family="linear",
            metrics={"latency_pytorch_ms": 2.10, "latency_triton_ms": 1.10, "speedup_x": 1.91},
        ),
    ]


def test_export_benchmark_table_handles_result_records_csv() -> None:
    """Verify export_benchmark_table handles ResultRecord dataclasses in CSV format."""
    records = _sample_records()
    csv_str = paper_export.export_benchmark_table(records, format="csv")
    assert csv_str
    reader = csv.reader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 3  # header + 2 rows
    assert "module_name" in rows[0] or "module" in rows[0]


def test_export_benchmark_table_handles_result_records_markdown() -> None:
    """Verify export_benchmark_table handles ResultRecord dataclasses in Markdown format."""
    records = _sample_records()
    md_str = paper_export.export_benchmark_table(records, format="markdown")
    assert md_str
    lines = md_str.strip().splitlines()
    assert len(lines) >= 4
    assert lines[0].startswith("|")
    assert "---" in lines[1]


def test_export_benchmark_table_handles_result_records_latex() -> None:
    """Verify export_benchmark_table handles ResultRecord dataclasses in LaTeX format."""
    records = _sample_records()
    tex_str = paper_export.export_benchmark_table(records, format="latex")
    assert tex_str
    assert r"\begin{table}" in tex_str
    assert r"\begin{tabular}" in tex_str
    assert r"\end{tabular}" in tex_str
    assert r"\end{table}" in tex_str
