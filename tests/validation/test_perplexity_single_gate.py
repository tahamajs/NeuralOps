#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying single source / distinctly documented perplexity gates and status-returning validators (F-57).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import neuralops.validation as validation
from neuralops.validation.gates.behavioral.perplexity import (
    PerplexityDeltaGate as BehavioralPerplexityDeltaGate,
    PerplexityGate as BehavioralPerplexityAlias,
)
from neuralops.validation.gates.empirical.perplexity import (
    EmpiricalPerplexityGate,
    PerplexityGate as EmpiricalPerplexityGateClass,
)


def test_single_or_distinct_perplexity_gates() -> None:
    """Verify L4 Empirical and L3 Behavioral perplexity gates are distinctly named and documented."""
    # L4 Empirical gate
    assert EmpiricalPerplexityGateClass.level == "L4"
    assert EmpiricalPerplexityGateClass.name == "perplexity"
    assert EmpiricalPerplexityGate is EmpiricalPerplexityGateClass

    # L3 Behavioral delta gate
    assert BehavioralPerplexityDeltaGate.level == "L3"
    assert BehavioralPerplexityDeltaGate.name == "perplexity_delta"
    assert BehavioralPerplexityAlias is BehavioralPerplexityDeltaGate

    # Both gates must document their scope distinction in their docstrings
    emp_doc = EmpiricalPerplexityGateClass.__doc__ or ""
    beh_doc = BehavioralPerplexityDeltaGate.__doc__ or ""

    assert "Scope Distinction" in emp_doc
    assert "L3 Behavioral" in emp_doc
    assert "Scope Distinction" in beh_doc
    assert "L4 Empirical" in beh_doc

    # Catalog in gates/__init__.py exposes both unambiguously
    from neuralops.validation.gates import PerplexityDeltaGate, PerplexityGate

    assert PerplexityGate is EmpiricalPerplexityGateClass
    assert PerplexityDeltaGate is BehavioralPerplexityDeltaGate


def test_validators_return_status_not_print(capsys) -> None:
    """Verify legacy validators return explicit status dictionaries and never print to stdout/stderr."""
    # 1. NumericalValidator
    t_normal = torch.randn(4, 4)
    res_range = validation.NumericalValidator.check_range(t_normal, "test_tensor", min_val=-10.0, max_val=10.0)
    assert isinstance(res_range, dict)
    assert "status" in res_range
    assert res_range["status"].lower() == "passed"

    res_finite = validation.NumericalValidator.check_finite(t_normal, "test_tensor")
    assert isinstance(res_finite, dict)
    assert "status" in res_finite
    assert res_finite["status"].lower() == "passed"

    # 2. SpeedValidator
    class DummyLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, input_ids):
            loss = self.linear(input_ids).sum()
            return type("Output", (), {"loss": loss})()

    dummy_model = DummyLinearModel()
    batch = {"input_ids": torch.randn(2, 8)}
    res_speed = validation.SpeedValidator.benchmark(dummy_model, batch, device="cpu", warmup=1, iters=2)
    assert isinstance(res_speed, dict)
    assert "status" in res_speed
    assert res_speed["status"] == "passed"

    # 3. TrainingValidator
    res_train = validation.TrainingValidator.overfit_test(dummy_model, batch, n_samples=2, steps=3, lr=1e-2)
    assert isinstance(res_train, dict)
    assert "status" in res_train
    assert res_train["status"] in ("passed", "failed")

    # Assert ZERO stdout or stderr was printed across all validator calls
    captured = capsys.readouterr()
    assert captured.out == "", f"Expected no stdout print output, got: {captured.out!r}"
    assert captured.err == "", f"Expected no stderr print output, got: {captured.err!r}"


def test_validation_init_exports() -> None:
    """Verify validation/__init__.py exports every imported symbol in __all__."""
    for name in validation.__all__:
        assert hasattr(validation, name), f"validation.__all__ references missing attribute: {name}"

    # Ensure recipe validation symbols are exported
    assert hasattr(validation, "validate_recipe_static")
    assert hasattr(validation, "SymplecticNumericalGuard")
    assert hasattr(validation, "E_W1_INVALID_SYNTAX")
