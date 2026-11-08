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
  1. Deterministic seeding reproducibility across random, numpy, and torch.
  2. Math-mode LaTeX standard deviation syntax compliance.
  3. Student's t and non-parametric bootstrap confidence interval estimation.
  4. Welch's two-sample t-test with Welch-Satterthwaite degrees of freedom and Cohen's d.
  5. Multi-seed orchestrator lifecycle, dry-run planning, and artifact serialization.
  6. CLI invocation for 'neuralops multiseed' and 'neuralops benchmark multiseed'.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from neuralops.benchmarks.cli import main
from neuralops.benchmarks.multi_seed import (
    DUAL_TIER_SEEDS,
    ENTERPRISE_5_SEEDS,
    TIER1_CANONICAL_SEEDS,
    TIER2_ADVERSARIAL_SEEDS,
    MultiSeedBenchmarkOrchestrator,
    MultiSeedConfig,
    compute_bootstrap_ci,
    compute_cohens_d,
    compute_students_t_ci,
    compute_welch_ttest,
    format_latex_sd,
    run_multiseed_benchmark,
    set_deterministic_seed,
)


def test_deterministic_seeding():
    """Verify that set_deterministic_seed enforces reproducible sequences across generators."""
    set_deterministic_seed(42)
    py_val1 = random.random()
    np_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()

    set_deterministic_seed(123)
    py_val_diff = random.random()
    assert py_val1 != py_val_diff

    # Reset back to 42
    set_deterministic_seed(42)
    assert random.random() == py_val1
    assert np.random.rand() == np_val1
    assert torch.rand(1).item() == torch_val1


def test_latex_standard_deviation_formatting():
    """Verify strict math-mode English standard deviation notation compliance."""
    assert format_latex_sd(86.40, 0.32, unit="%") == r"$86.40 \pm 0.32\%$"
    assert format_latex_sd(1.24, 0.03, unit="x") == r"$1.24 \pm 0.03\times$"
    assert format_latex_sd(1.24, 0.03, unit=r"\times") == r"$1.24 \pm 0.03\times$"
    assert format_latex_sd(4.12, 0.04, unit="ms") == r"$4.12 \pm 0.04\text{ ms}$"
    assert format_latex_sd(142.15, 1.20, unit="GB/s") == r"$142.15 \pm 1.20\text{ GB/s}$"
    assert format_latex_sd(27.33, 0.07, unit="") == r"$27.33 \pm 0.07$"


def test_statistical_confidence_intervals():
    """Verify parametric Student's t and non-parametric bootstrap confidence interval estimation."""
    # Constant sample
    const_samples = [5.0, 5.0, 5.0, 5.0]
    ci_t_const = compute_students_t_ci(const_samples)
    assert ci_t_const == (5.0, 5.0)

    ci_boot_const = compute_bootstrap_ci(const_samples)
    assert ci_boot_const == (5.0, 5.0)

    # Distributed sample
    samples = [10.0, 12.0, 9.0, 11.0, 10.5, 11.5]
    ci_t = compute_students_t_ci(samples, alpha=0.05)
    mean = float(np.mean(samples))
    assert ci_t[0] < mean < ci_t[1]

    ci_boot = compute_bootstrap_ci(samples, n_boot=2000, ci=0.95, seed=42)
    assert ci_boot[0] < mean < ci_boot[1]


def test_welch_two_sample_ttest_and_cohens_d():
    """Verify Welch's unequal-variance t-test and Cohen's d effect size."""
    # Proposed system vs baseline system
    proposed = [100.0, 102.0, 101.5, 103.0, 101.0]
    baseline = [90.0, 91.0, 89.5, 92.0, 90.5]

    res = compute_welch_ttest(proposed, baseline, alpha=0.05)
    assert res["method"] == "Welch's two-sample t-test"
    assert res["difference"] > 0
    assert res["t_statistic"] > 0
    assert res["p_value"] < 0.001
    assert res["significant"] is True
    assert res["cohens_d"] > 2.0  # Very large positive effect size

    # Identical samples
    res_ident = compute_welch_ttest(proposed, proposed)
    assert res_ident["p_value"] == 1.0
    assert res_ident["difference"] == 0.0
    assert res_ident["cohens_d"] == 0.0


def test_multiseed_orchestrator_dry_run(tmp_path: Path):
    """Verify MultiSeedBenchmarkOrchestrator dry run planning, serialization, and LaTeX generation."""
    config = MultiSeedConfig(
        modules=["attention:full_attention", "attention:cosformer"],
        preset="canonical3",
        baseline_module="attention:full_attention",
        output_dir=str(tmp_path / "multiseed_test"),
        dry_run=True,
    )
    orchestrator = MultiSeedBenchmarkOrchestrator(config)
    result = orchestrator.run()

    assert len(result.module_results) == 2
    assert result.module_results["attention:full_attention"].status == "PLANNED"
    assert len(result.config.seeds) == 3

    # Minimal real run on CPU to test execution, serialization, and LaTeX generation
    real_config = MultiSeedConfig(
        modules=["attention:full_attention", "attention:cosformer"],
        seeds=[42, 123],
        baseline_module="attention:full_attention",
        seq_len=64,
        hidden_dim=32,
        warmup=1,
        iters=2,
        output_dir=str(tmp_path / "multiseed_real"),
        dry_run=False,
    )
    real_orch = MultiSeedBenchmarkOrchestrator(real_config)
    real_result = real_orch.run()

    assert len(real_result.module_results) == 2
    cos_res = real_result.module_results["attention:cosformer"]
    assert cos_res.status == "SUCCESS"
    assert "latency_ms" in cos_res.metrics
    assert len(cos_res.seed_measurements) == 2
    assert cos_res.baseline_comparison is not None

    # Check LaTeX table generation
    latex_table = real_result.to_latex_table()
    assert r"\begin{table}[t]" in latex_table
    assert r"\caption{Multi-Seed Statistical Significance Evaluation" in latex_table
    assert r"\bottomrule" in latex_table

    # Check serialization
    saved_path = real_result.save()
    assert saved_path.exists()
    assert saved_path.with_suffix(".tex").exists()


def test_multiseed_presets():
    """Verify seed presets taxonomy."""
    assert len(TIER1_CANONICAL_SEEDS) == 3
    assert len(TIER2_ADVERSARIAL_SEEDS) == 3
    assert len(DUAL_TIER_SEEDS) == 6
    assert len(ENTERPRISE_5_SEEDS) == 5

    res = run_multiseed_benchmark(
        modules=["attention:full_attention"],
        preset="enterprise5",
        dry_run=True,
    )
    assert len(res.config.seeds) == 5
    assert res.config.seeds == list(ENTERPRISE_5_SEEDS)


def test_cli_multiseed_subcommand():
    """Verify CLI integration of 'neuralops multiseed' and 'neuralops benchmark multiseed'."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        exit_code = main([
            "multiseed",
            "--modules",
            "attention:full_attention,attention:cosformer",
            "--preset",
            "canonical3",
            "--plan",
        ])
    assert exit_code == 0
    assert "[✓] Multi-Seed Benchmark Plan: 2 module(s) across 3 seed(s)." in buf.getvalue()

    # Test benchmark subcommand syntax
    buf_bench = io.StringIO()
    with patch("sys.stdout", buf_bench):
        exit_code_bench = main([
            "benchmark",
            "multiseed",
            "--modules",
            "attention:full_attention",
            "--preset",
            "canonical3",
            "--plan",
            "--latex",
        ])
    assert exit_code_bench == 0
    assert r"\begin{table}[t]" in buf_bench.getvalue()
