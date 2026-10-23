#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Gradient-Aware Calibration (P040).

Verifies that backward passes in calibration execute under active autograd
and do not raise RuntimeError, including when invoked under torch.no_grad().
"""

import pytest
import torch
import torch.nn as nn

from neuralops.surgery.calibration import (
    GradientCalibrator,
    calibrate_gradient_importance,
)
from neuralops.surgery.universal_calibrator import UniversalActivationCalibrator


class MockLinearBackbone(nn.Module):
    def __init__(self, in_features: int = 16, out_features: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_calibrator_gradient_backward_executes_without_error():
    """Verify that GradientCalibrator runs backward without RuntimeError."""
    model = MockLinearBackbone(in_features=16, out_features=16)
    calibrator = GradientCalibrator(model=model)

    batches = [torch.randn(4, 16) for _ in range(3)]
    results = calibrator.calibrate(batches)

    assert "fc1" in results
    assert "fc2" in results
    assert "grad_energy" in results["fc1"]
    assert "importance" in results["fc1"]
    assert results["fc1"]["grad_energy"].item() >= 0.0
    assert results["fc2"]["importance"].item() >= 0.0


def test_calibrator_backward_under_caller_no_grad_block():
    """
    P040 Regression Test: In PyTorch 2.1+, calling backward() inside a no_grad
    block raises RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn.
    The calibrator must shield internal loss computation and backward with enable_grad.
    """
    model = MockLinearBackbone(in_features=16, out_features=16)
    calibrator = GradientCalibrator(model=model)

    batches = [torch.randn(4, 16) for _ in range(2)]

    # Caller explicitly wraps invocation in no_grad
    with torch.no_grad():
        results = calibrator.calibrate(batches)

    assert results is not None
    assert "fc1" in results
    assert results["fc1"]["grad_energy"].item() >= 0.0


def test_calibrate_gradient_importance_convenience_function():
    """Test functional wrapper calibrate_gradient_importance."""
    model = MockLinearBackbone(in_features=8, out_features=8)
    batch = torch.randn(2, 8)

    results = calibrate_gradient_importance(model, batch)
    assert len(results) > 0
    assert "fc1" in results


def test_universal_calibrator_gradient_mode():
    """Test compute_gradients=True in UniversalActivationCalibrator."""
    model = MockLinearBackbone(in_features=16, out_features=16)
    calibrator = UniversalActivationCalibrator(model=model)

    batches = [torch.randn(4, 16) for _ in range(2)]
    with torch.no_grad():
        results = calibrator.calibrate(batches, compute_gradients=True)

    assert "fc1" in results
    assert "hessian" in results["fc1"]
