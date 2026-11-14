#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for EmpiricalBenchmarkGate honest NotImplemented behavior (P0-7).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from neuralops.validation.profiles import EmpiricalBenchmarkGate


def test_empirical_gate_skipped_without_dataset() -> None:
    gate = EmpiricalBenchmarkGate()
    model = nn.Linear(4, 4)

    # Context is None
    res = gate.run(model=model, context=None)
    assert res.skipped is True
    assert res.passed is True
    assert "not configured" in res.message

    # Context without eval_dataset
    res2 = gate.run(model=model, context={"device": "cpu"})
    assert res2.skipped is True
    assert res2.passed is True


def test_empirical_gate_raises_honest_not_implemented_with_dataset() -> None:
    gate = EmpiricalBenchmarkGate()
    model = nn.Linear(4, 4)

    with pytest.raises(NotImplementedError) as exc_info:
        gate.run(model=model, context={"eval_dataset": "synthetic_bench"})

    assert "EmpiricalBenchmarkGate evaluation is not implemented yet" in str(exc_info.value)
    assert "empirical evaluation must be performed through the benchmark suite" in str(exc_info.value)
