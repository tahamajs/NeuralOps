#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Multi-Seed Statistical Significance & Cold-Cache Microbenchmarking (AGT-60).

Verifies:
  1. Multi-seed randomized execution across standard preset tiers.
  2. Statistical estimator calculations (Student's t CI, bootstrap CI, Welch's t-test, Cohen's d).
  3. LaTeX table serialization with publication-grade formatting.
  4. CLI multiseed subparser invocation in plan and JSON modes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from neuralops.benchmarks.cli import main
from neuralops.benchmarks.multi_seed import (
    MultiSeedBenchmarkOrchestrator,
    MultiSeedConfig,
    compute_bootstrap_ci,
    compute_cohens_d,
    compute_students_t_ci,
    compute_welch_ttest,
    format_latex_sd,
    run_multiseed_benchmark,
)


def test_statistical_estimators():
    """Verify Student's t CI, bootstrap CI, Welch's t-test, and Cohen's d math."""
    data_a = [10.0, 10.2, 9.8, 10.1, 9.9]
    mean_a = sum(data_a) / len(data_a)
    assert abs(mean_a - 10.0) < 1e-4

    ci_t = compute_students_t_ci(data_a)
    assert ci_t[0] < mean_a < ci_t[1]

    ci_boot = compute_bootstrap_ci(data_a)
    assert ci_boot[0] < mean_a < ci_boot[1]

    latex_str = format_latex_sd(mean_a, 0.15, unit="ms")
    assert "10.00" in latex_str
    assert "0.15" in latex_str

    data_b = [15.0, 15.2, 14.8, 15.1, 14.9]
    welch = compute_welch_ttest(data_b, data_a)
    assert welch["p_value"] < 0.001
    assert welch["significant"] is True

    cohen = compute_cohens_d(data_b, data_a)
    assert cohen > 2.0  # Huge effect size


def test_multiseed_plan_and_execution_cpu(tmp_path: Path):
    """Verify dry-run planning and minimal CPU execution across seeds."""
    # Dry run
    plan_res = run_multiseed_benchmark(
        modules=["attention:full_attention"],
        seeds=[42, 123],
        dry_run=True,
    )
    assert len(plan_res.config.seeds) == 2
    assert plan_res.module_results["attention:full_attention"].status == "PLANNED"

    # Minimal real run on CPU
    run_res = run_multiseed_benchmark(
        modules=["attention:full_attention"],
        seeds=[42, 137],
        warmup=1,
        iters=2,
        seq_len=64,
        hidden_dim=32,
        batch_size=1,
        output_dir=str(tmp_path / "multiseed_out"),
        dry_run=False,
    )
    assert "attention:full_attention" in run_res.module_results
    mod_res = run_res.module_results["attention:full_attention"]
    assert "latency_ms" in mod_res.metrics
    assert len(mod_res.seed_measurements) == 2

    # Check LaTeX table generation
    latex = run_res.to_latex_table()
    assert "begin{tabular}" in latex
    assert "attention:full_attention" in latex


def test_multiseed_cli_dry_run():
    """Verify 'neuralops benchmark multiseed --dry-run' via CLI."""
    with patch("sys.argv", ["neuralops", "benchmark", "multiseed", "--dry-run"]):
        assert main() == 0


def test_multiseed_cli_json(capsys, tmp_path: Path):
    """Verify 'neuralops benchmark multiseed --json' output format."""
    with patch("sys.argv", [
        "neuralops",
        "benchmark",
        "multiseed",
        "--modules",
        "attention:full_attention",
        "--seeds",
        "42",
        "--warmup",
        "1",
        "--iters",
        "2",
        "--seq-len",
        "64",
        "--hidden-dim",
        "32",
        "--output-dir",
        str(tmp_path / "cli_json"),
        "--json",
    ]):
        code = main()
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "config" in data
        assert "module_results" in data
