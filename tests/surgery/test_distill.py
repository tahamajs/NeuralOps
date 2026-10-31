#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for distill_from_teacher surgery recovery operator.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.surgery.recovery.distill import distill_from_teacher
from neuralops.surgery.registry import get_surgery


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        return self.fc(x)


def test_distill_metadata():
    meta = get_surgery("distill_from_teacher")
    assert meta.name == "distill_from_teacher"
    assert meta.category == "recovery"
    assert meta.requires_gpu is True
    assert meta._experimental is True


def test_distill_execution():
    student = SimpleToyModel()
    teacher = SimpleToyModel()
    x = torch.randn(2, 32)

    with pytest.warns(UserWarning, match="experimental"):
        healed = distill_from_teacher(
            student,
            teacher=teacher,
            steps=10,
            pillars=["logit", "relation", "operator", "cka"],
        )

    out = healed(x)
    assert out.shape == (2, 32)
