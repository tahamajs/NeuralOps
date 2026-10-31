#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for L4 Empirical Validation Gates (Sub-Agent C2).
"""

import pytest
import torch
import torch.nn as nn

from neuralops.validation.gates.empirical import (
    AccuracyGate,
    LatencyGate,
    NeedleInAHaystackGate,
    ThroughputGate,
    VRAMGate,
)


class SimpleMLP(nn.Module):

    def __init__(self, in_features: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_c2_empirical_gates_skipped_on_cpu_without_data():
    """Verify all L4 gates return skipped=True when resources are absent without fabrication."""
    model = SimpleMLP()

    # Accuracy without dataset/eval_fn
    acc_gate = AccuracyGate()
    acc_res = acc_gate.run(model=model, context={})
    assert acc_res.passed is True
    assert acc_res.skipped is True
    assert "No dataset or eval_fn provided" in acc_res.message

    # VRAM on CPU without CUDA
    vram_gate = VRAMGate()
    vram_res = vram_gate.run(model=model, context={})
    if not torch.cuda.is_available():
        assert vram_res.passed is True
        assert vram_res.skipped is True
        assert "CUDA is not available" in vram_res.message

    # NIAH without dataset
    niah_gate = NeedleInAHaystackGate()
    niah_res = niah_gate.run(model=model, context={})
    assert niah_res.passed is True
    assert niah_res.skipped is True
    assert "No NIAH dataset or niah_eval_fn provided" in niah_res.message


def test_c2_accuracy_gate_runs_on_mini_dataset():
    """Verify AccuracyGate runs on a toy classifier with a mini-dataset and reports real accuracy."""
    torch.manual_seed(42)
    model = SimpleMLP(in_features=8, num_classes=2)

    # Synthetic mini-dataset: 20 samples of 8-dim features and binary labels
    x = torch.randn(20, 8)
    y = torch.randint(0, 2, (20,))
    dataset = [(x, y)]

    gate = AccuracyGate(min_accuracy=0.0)
    res = gate.run(model=model, context={"dataset": dataset})

    assert res.skipped is False
    assert res.passed is True
    assert isinstance(res.measured_value, float)
    assert 0.0 <= res.measured_value <= 1.0


def test_c2_latency_gate_measures_real_cpu_latency():
    """Verify LatencyGate measures non-zero real CPU latency."""
    model = SimpleMLP(in_features=16, num_classes=2)
    gate = LatencyGate(warmup_runs=1, measure_runs=3)

    res = gate.run(model=model, context={"batch_size": 2, "seq_len": 16})

    assert res.skipped is False
    assert res.passed is True
    assert isinstance(res.measured_value, float)
    assert res.measured_value > 0.0
    assert "median_latency_ms" in res.details


def test_c2_throughput_gate_measures_real_throughput():
    """Verify ThroughputGate measures non-zero real CPU throughput."""
    model = SimpleMLP(in_features=16, num_classes=2)
    gate = ThroughputGate(warmup_runs=1, measure_runs=3)

    res = gate.run(model=model, context={"batch_size": 2, "seq_len": 16})

    assert res.skipped is False
    assert res.passed is True
    assert isinstance(res.measured_value, float)
    assert res.measured_value > 0.0
    assert "tokens_per_second" in res.details


def test_c2_vram_gate_cpu_vs_gpu():
    """Verify VRAMGate returns skipped=True on CPU and measures only when CUDA available."""
    model = SimpleMLP()
    gate = VRAMGate()
    res = gate.run(model=model)

    if not torch.cuda.is_available():
        assert res.skipped is True
        assert res.passed is True
        assert "zero-fabrication policy" in res.message
