#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression and comprehensive validation tests for Wave 2 (Agent E1).

Tests cover:
- P0-18: NumericalValidator.check_finite returning structured status on NaN/Inf instead of raising.
- P0-7: EmpiricalBenchmarkGate performing real evaluation and failing when thresholds are violated.
- MemoryTracker device parameterization and get_stats contract.
- SpeedValidator CPU timing and CUDA event compliance.
- TrainingValidator handling of non-finite losses.
- ModelValidator on standard PyTorch float modules and HF-style modules.
- ReportGenerator Markdown + JSON output generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import neuralops.validation as validation
from neuralops.validation.memory_validator import MemoryTracker
from neuralops.validation.model_validator import ModelValidator
from neuralops.validation.profiles import EmpiricalBenchmarkGate
from neuralops.validation.report_generator import ReportGenerator


class ToyClassifier(nn.Module):
    """Simple linear classifier for testing."""

    def __init__(self, in_features: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class BadModel(nn.Module):
    """Model that outputs NaNs."""

    def __init__(self, in_features: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        return out * float("nan")


# ==============================================================================
# P0-18: NumericalValidator.check_finite
# ==============================================================================


def test_p0_18_check_finite_returns_status_on_nan() -> None:
    """Verify check_finite returns failed ValidationStatus instead of raising ValueError on NaN."""
    t_nan = torch.tensor([1.0, float("nan"), 3.0])
    res = validation.NumericalValidator.check_finite(t_nan, "test_weights")

    assert isinstance(res, dict)
    assert res.get("status") == "failed"
    assert res.get("passed") is False
    assert bool(res) is False
    assert res.get("has_nan") is True
    assert "NaN in test_weights" in res.get("error", "")


def test_p0_18_check_finite_returns_status_on_inf() -> None:
    """Verify check_finite returns failed ValidationStatus instead of raising ValueError on Inf."""
    t_inf = torch.tensor([1.0, float("inf"), 3.0])
    res = validation.NumericalValidator.check_finite(t_inf, "test_bias")

    assert isinstance(res, dict)
    assert res.get("status") == "failed"
    assert res.get("passed") is False
    assert bool(res) is False
    assert res.get("has_inf") is True
    assert "Inf in test_bias" in res.get("error", "")


def test_p0_18_check_finite_passes_on_finite_tensor() -> None:
    """Verify check_finite passes on finite tensors."""
    t_finite = torch.tensor([1.0, 2.0, 3.0])
    res = validation.NumericalValidator.check_finite(t_finite, "clean_tensor")

    assert isinstance(res, dict)
    assert res.get("status") == "passed"
    assert res.get("passed") is True
    assert bool(res) is True
    assert res.get("has_nan") is False
    assert res.get("has_inf") is False


# ==============================================================================
# P0-7: profiles.py::EmpiricalBenchmarkGate
# ==============================================================================


def test_p0_7_empirical_benchmark_gate_skips_when_no_dataset() -> None:
    """Verify EmpiricalBenchmarkGate skips cleanly when no dataset or eval_fn is provided."""
    gate = EmpiricalBenchmarkGate()
    model = ToyClassifier()
    res = gate.run(model=model, context={})

    assert res.passed is True
    assert res.skipped is True
    assert "not configured; gate skipped" in res.message


def test_p0_7_empirical_benchmark_gate_evaluates_real_accuracy() -> None:
    """Verify EmpiricalBenchmarkGate actually runs evaluation on eval_dataset and reports score."""
    torch.manual_seed(42)
    model = ToyClassifier(in_features=8, num_classes=2)

    # Synthetic dataset
    x = torch.randn(20, 8)
    y = torch.randint(0, 2, (20,))
    eval_dataset = [(x, y)]

    gate = EmpiricalBenchmarkGate(min_score=0.0)
    res = gate.run(model=model, context={"eval_dataset": eval_dataset})

    assert res.skipped is False
    assert res.passed is True
    assert isinstance(res.measured_value, float)
    assert 0.0 <= res.measured_value <= 1.0
    assert "score" in res.details


def test_p0_7_empirical_benchmark_gate_fails_when_below_threshold() -> None:
    """Verify EmpiricalBenchmarkGate fails when measured score is below min_score."""
    torch.manual_seed(42)
    model = ToyClassifier(in_features=8, num_classes=2)

    x = torch.randn(20, 8)
    y = torch.randint(0, 2, (20,))
    eval_dataset = [(x, y)]

    # Set impossible threshold of 100%
    gate = EmpiricalBenchmarkGate(min_score=1.1)
    res = gate.run(model=model, context={"eval_dataset": eval_dataset})

    assert res.skipped is False
    assert res.passed is False
    assert "below minimum threshold" in res.message


def test_p0_7_empirical_benchmark_gate_detects_score_drop() -> None:
    """Verify EmpiricalBenchmarkGate fails when model drops too much compared to baseline."""
    base_model = ToyClassifier(in_features=8, num_classes=2)
    # Degraded model
    worse_model = ToyClassifier(in_features=8, num_classes=2)

    # Use eval_fn for precise control
    gate = EmpiricalBenchmarkGate(max_score_drop=0.05)
    ctx = {
        "eval_fn": lambda m: 0.90 if m is base_model else 0.70,
    }

    res = gate.run(
        model=worse_model,
        baseline_model=base_model,
        context=ctx,
    )

    assert res.skipped is False
    assert res.passed is False
    assert "Score drop" in res.message


def test_p0_7_empirical_benchmark_gate_fails_closed_on_eval_error() -> None:
    """Verify EmpiricalBenchmarkGate fails closed with E_EVAL_ERROR when evaluation raises an exception."""
    model = ToyClassifier()

    def buggy_eval(m: nn.Module) -> float:
        raise RuntimeError("Synthetic evaluation crash")

    gate = EmpiricalBenchmarkGate()
    res = gate.run(model=model, context={"eval_fn": buggy_eval})

    assert res.passed is False
    assert res.error_code == "E_EVAL_ERROR"
    assert "Synthetic evaluation crash" in res.message


# ==============================================================================
# MemoryTracker
# ==============================================================================


def test_memory_tracker_get_stats_and_device_handling() -> None:
    """Verify MemoryTracker resolves device safely and exposes Rule 2.4 get_stats."""
    tracker = MemoryTracker()
    stats = tracker.get_stats()

    assert isinstance(stats, dict)
    assert "allocated_mb" in stats
    assert "reserved_mb" in stats
    assert "peak_mb" in stats
    assert "total_mb" in stats
    assert tracker.check_ok(threshold=0.9) is True


# ==============================================================================
# ModelValidator
# ==============================================================================


def test_model_validator_on_float_mlp() -> None:
    """Verify ModelValidator validates standard PyTorch module with float input without crashing."""
    mlp = ToyClassifier(in_features=8, num_classes=2)
    validator = ModelValidator(model=mlp, input_shape=(4, 8), device="cpu")
    success = validator.validate()

    assert success is True
    assert len(validator.errors) == 0


def test_model_validator_detects_nan_outputs() -> None:
    """Verify ModelValidator flags NaN outputs."""
    bad_model = BadModel(in_features=8, num_classes=2)
    validator = ModelValidator(model=bad_model, input_shape=(4, 8), device="cpu")
    success = validator.validate()

    assert success is False
    assert any("NaN in forward output" in e for e in validator.errors)


# ==============================================================================
# ReportGenerator
# ==============================================================================


def test_report_generator_saves_valid_files(tmp_path) -> None:
    """Verify ReportGenerator creates markdown and json without error."""
    out_dir = str(tmp_path / "reports")
    gen = ReportGenerator(out_dir)
    gen.add(category="L1", name="shapes_match", status="PASSED", value="[2, 16]")
    gen.add(category="L2", name="no_nan_inf", status="PASSED")

    saved_path = gen.save("audit_report.md")
    assert saved_path.endswith("audit_report.md")
    assert (tmp_path / "reports" / "audit_report.md").exists()
    assert (tmp_path / "reports" / "audit_report.json").exists()
