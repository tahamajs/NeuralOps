#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Sub-Agent B3: Basic Validation Agent (L1 Structural & L2 Numerical Gates).
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from neuralops.validation.gates.numerical.forward_pass import ForwardPassRunsGate
from neuralops.validation.gates.numerical.no_nan_inf import NoNanInfGate
from neuralops.validation.gates.structural.connectivity import ConnectivityGate
from neuralops.validation.gates.structural.dtypes import DtypesMatchGate
from neuralops.validation.gates.structural.shapes import ShapesMatchGate
from neuralops.validation.profiles import ValidationProfile
from neuralops.validation.runner import run_validation_gates


class SimpleDummyNet(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_b3_l1_gates_pass_on_valid_pair():
    """Verify each L1 gate returns GateResult(passed=True) for a valid pair."""
    m_before = SimpleDummyNet()
    m_after = SimpleDummyNet()

    res_shapes = ShapesMatchGate().run(model_before=m_before, model_after=m_after)
    assert res_shapes.passed is True
    assert res_shapes.gate_name == "shapes_match"

    res_dtypes = DtypesMatchGate().run(model_before=m_before, model_after=m_after)
    assert res_dtypes.passed is True
    assert res_dtypes.gate_name == "dtypes_match"

    res_conn = ConnectivityGate().run(model_before=m_before, model_after=m_after)
    assert res_conn.passed is True
    assert res_conn.gate_name == "connectivity"


def test_b3_l1_connectivity_gate_fails_on_orphan_module():
    """Verify L1 connectivity gate fails for a model with an orphan module."""
    model = SimpleDummyNet()
    # Attach an unlinked orphan module directly in __dict__ without registering in _modules
    model.__dict__["orphan_layer"] = nn.Linear(10, 10)

    res = ConnectivityGate().run(model=model)
    assert res.passed is False
    assert "Orphan module" in res.message or "mismatch" in res.message


def test_b3_l2_no_nan_inf_gate_detects_nan():
    """Verify L2 no_nan_inf gate fails when NaN is injected into a weight."""
    model = SimpleDummyNet()
    with torch.no_grad():
        model.fc1.weight[0, 0] = float("nan")

    res = NoNanInfGate().run(model=model)
    assert res.passed is False
    assert "NaN values" in res.message


def test_b3_l2_forward_pass_runs():
    """Verify L2 forward_pass gate runs the model on a small batch and returns passed=True."""
    model = SimpleDummyNet()
    res = ForwardPassRunsGate().run(model=model)
    assert res.passed is True
    assert "completed successfully" in res.message


def test_b3_profile_structural_runs_only_l1_gates():
    """Verify run_validation_gates with profile=STRUCTURAL runs only L1 gates (not L2)."""
    m_before = SimpleDummyNet()
    m_after = SimpleDummyNet()

    val_res = run_validation_gates(
        model_before=m_before,
        model_after=m_after,
        profile=ValidationProfile.STRUCTURAL,
    )
    assert val_res.passed is True
    gate_names = [r.gate_name for r in val_res.gate_results]
    assert "shapes_match" in gate_names
    assert "dtypes_match" in gate_names
    assert "connectivity" in gate_names
    # L2 gates must NOT be run in STRUCTURAL profile
    assert "forward_pass_runs" not in gate_names
    assert "no_nan_inf" not in gate_names


def test_b3_profile_production_runs_all_gates_and_skips_unavailable_empirical():
    """Verify run_validation_gates with profile=PRODUCTION runs all gates and marks unavailable empirical skipped=True."""
    import copy
    m_before = SimpleDummyNet()
    m_after = copy.deepcopy(m_before)

    val_res = run_validation_gates(
        model_before=m_before,
        model_after=m_after,
        profile=ValidationProfile.PRODUCTION,
    )
    assert val_res.passed is True
    gate_names = [r.gate_name for r in val_res.gate_results]
    assert "shapes_match" in gate_names
    assert "forward_pass_runs" in gate_names
    assert "empirical_benchmark" in gate_names

    # Empirical gate skipped because no dataset configured in context
    empirical_res = next(r for r in val_res.gate_results if r.gate_name == "empirical_benchmark")
    assert empirical_res.skipped is True
    assert empirical_res.passed is True


def test_b3_validation_result_json_roundtrip():
    """Verify ValidationResult.to_dict() is JSON-serializable."""
    m_before = SimpleDummyNet()
    m_after = SimpleDummyNet()

    val_res = run_validation_gates(
        model_before=m_before,
        model_after=m_after,
        profile=ValidationProfile.NUMERICAL,
    )
    res_dict = val_res.to_dict()
    json_str = json.dumps(res_dict)
    loaded = json.loads(json_str)
    assert loaded["passed"] is True
    assert len(loaded["gates"]) == len(val_res.gate_results)
