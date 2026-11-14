#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for NumericalValidator.check_finite returning ValidationStatus without raising (P0-18).
"""

from __future__ import annotations

import torch

from neuralops.validation.numerical_validator import NumericalValidator
from neuralops.validation.training_validator import ValidationStatus


def test_check_finite_on_finite_tensor() -> None:
    t = torch.randn(8, 8)
    res = NumericalValidator.check_finite(t, "weights")

    assert isinstance(res, ValidationStatus)
    assert res.passed is True
    assert bool(res) is True
    assert res["status"] == "passed"
    assert res["nan_count"] == 0
    assert res["inf_count"] == 0
    assert res["name"] == "weights"


def test_check_finite_on_nan_tensor() -> None:
    t = torch.randn(8, 8)
    t[0, 0] = float("nan")
    t[2, 3] = float("nan")

    # Must NOT raise ValueError
    res = NumericalValidator.check_finite(t, "activations")

    assert isinstance(res, ValidationStatus)
    assert res.passed is False
    assert bool(res) is False
    assert res["status"] == "failed"
    assert res["nan_count"] == 2
    assert res["inf_count"] == 0
    assert res["name"] == "activations"


def test_check_finite_on_inf_tensor() -> None:
    t = torch.randn(8, 8)
    t[1, 1] = float("inf")
    t[3, 3] = float("-inf")

    # Must NOT raise ValueError
    res = NumericalValidator.check_finite(t, "gradients")

    assert isinstance(res, ValidationStatus)
    assert res.passed is False
    assert bool(res) is False
    assert res["status"] == "failed"
    assert res["nan_count"] == 0
    assert res["inf_count"] == 2
    assert res["name"] == "gradients"


def test_check_finite_on_mixed_nan_and_inf() -> None:
    t = torch.tensor([1.0, float("nan"), float("inf"), float("-inf"), 2.0])
    res = NumericalValidator.check_finite(t, "mixed_tensor")

    assert isinstance(res, ValidationStatus)
    assert res.passed is False
    assert bool(res) is False
    assert res["status"] == "failed"
    assert res["nan_count"] == 1
    assert res["inf_count"] == 2
