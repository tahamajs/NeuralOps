#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for publication-grade LaTeX table generation and headless CLI degradation.
"""

from __future__ import annotations

import io
import sqlite3
import sys
from unittest.mock import patch

import pandas as pd

from neuralops.cli.bench_cmd import _render_results_table
from neuralops.cli.terminal_ui import print_banner, print_hf_style_help, run_interactive_ui
from neuralops.reporting.latex_table_generator import LatexTableGenerator


def test_latex_table_generator_from_dataframe():
    """Verify IEEE/ACM booktabs formatting from a pandas DataFrame."""
    df = pd.DataFrame({
        "Model": ["GPT2-Base", "LLaMA-7B"],
        "Attention": ["MHA", "KDA"],
        "Throughput (tok/s)": [1240.5, 3420.8],
        "Speedup": [1.0, 2.76],
    })
    generator = LatexTableGenerator(commit_sha="a1b2c3d4", hardware_target="NVIDIA RTX 3090 (24GB)")
    latex_code = generator.generate_from_dataframe(
        df,
        caption="Efficiency Benchmark Across Architectures",
        label="tab:attn_efficiency",
    )

    assert r"\begin{table}[t]" in latex_code
    assert r"\centering" in latex_code
    assert r"\small" in latex_code
    assert r"\caption{Efficiency Benchmark Across Architectures}" in latex_code
    assert r"\label{tab:attn_efficiency}" in latex_code
    assert r"\begin{tabular}{lccc}" in latex_code
    assert r"\toprule" in latex_code
    assert r"\textbf{Model} & \textbf{Attention} & \textbf{Throughput (tok/s)} & \textbf{Speedup} \\" in latex_code
    assert r"\midrule" in latex_code
    assert "GPT2-Base & MHA & 1240.50 & 1.00 \\\\" in latex_code
    assert "LLaMA-7B & KDA & 3420.80 & 2.76 \\\\" in latex_code
    assert r"\bottomrule" in latex_code
    assert r"\end{tabular}" in latex_code
    assert r"\end{table}" in latex_code
    assert "% Git Commit SHA: a1b2c3d4" in latex_code
    assert "% Hardware Target: NVIDIA RTX 3090 (24GB)" in latex_code


def test_latex_table_generator_from_sqlite(tmp_path):
    """Verify extraction and generation directly from an SQLite database."""
    db_file = tmp_path / "master_ledger.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE benchmark_runs (run_id TEXT, model_name TEXT, speedup REAL, status TEXT)"
    )
    cur.execute(
        "INSERT INTO benchmark_runs VALUES ('r1', 'gpt2', 1.45, 'success')"
    )
    cur.execute(
        "INSERT INTO benchmark_runs VALUES ('r2', 'llama2', 2.15, 'success')"
    )
    conn.commit()
    conn.close()

    generator = LatexTableGenerator()
    latex = generator.generate_from_sqlite(
        str(db_file),
        "SELECT model_name AS Model, speedup AS Speedup FROM benchmark_runs WHERE status='success'",
        caption="Selected Benchmark Runs",
        label="tab:selected_runs",
    )

    assert r"\caption{Selected Benchmark Runs}" in latex
    assert r"\label{tab:selected_runs}" in latex
    assert "gpt2 & 1.45 \\\\" in latex
    assert "llama2 & 2.15 \\\\" in latex


def test_headless_non_tty_fallback(tmp_path, capsys):
    """Verify that CLI components fall back gracefully without Rich errors in non-TTY (PBS job) mode."""
    # 1. Test terminal_ui banner in headless mode
    with patch("neuralops.cli.terminal_ui.HAS_RICH", False):
        print_banner()
        out = capsys.readouterr().out
        assert "NeuralOps CLI" in out

    # 2. Test terminal_ui help in headless mode
    with patch("neuralops.cli.terminal_ui.HAS_RICH", False):
        print_hf_style_help()
        out = capsys.readouterr().out
        assert "Usage: neuralops" in out
        assert "Commands:" in out

    # 3. Test interactive UI without rich falling back to help
    with patch("neuralops.cli.terminal_ui.HAS_RICH", False):
        run_interactive_ui()
        out = capsys.readouterr().out
        assert "Interactive mode requires 'rich'. Falling back to standard help." in out

    # 4. Test bench_cmd _render_results_table plain-text fallback
    journal_path = tmp_path / ".journal.jsonl"
    journal_path.write_text(
        '{"status": "success", "model_name": "gpt2", "transplanted_attention": "kda", "benchmark_suite": "eval", "metrics": {"acc": 0.95}}\n'
    )
    with patch("neuralops.cli.bench_cmd.HAS_RICH", False):
        _render_results_table(tmp_path)
        out = capsys.readouterr().out
        assert "=== Benchmark Matrix Summary ===" in out
        assert "gpt2 | kda" in out
