#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for reproducibility benchmark gate (finding F-94).
"""

from __future__ import annotations

import pytest
from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.reproducibility import run_reproducibility


def test_reproducibility_gate_cpu(tmp_path):
    """Verify reproducibility benchmark executes on CPU and confirms reproducibility."""
    cfg = BenchmarkConfig(
        device="cpu",
        seed=42,
        output_dir=tmp_path,
    )
    res = run_reproducibility(cfg)
    assert res.status == "success"
    assert res.metrics["reproducible"] is True
    assert res.metrics["exact_match"] is True
    assert res.metrics["max_diff"] == 0.0
