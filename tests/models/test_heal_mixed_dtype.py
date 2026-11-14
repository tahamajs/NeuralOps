#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for _heal_replaced_model_impl under mixed-dtype parameters."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuralops.models.surgery import _heal_replaced_model_impl


class MixedDtypeLinear(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        # Parameter 1 in float32
        self.w1 = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32))
        # Parameter 2 in float16
        self.w2 = nn.Parameter(torch.randn(dim, dim, dtype=torch.float16))

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        # Performs linear algebra with its parameters
        h = torch.matmul(x.to(self.w1.dtype), self.w1)
        return torch.matmul(h.to(self.w2.dtype), self.w2).to(x.dtype)


class SimpleTeacherBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)


class MixedStudentBlock(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.self_attn = MixedDtypeLinear(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)


def test_heal_replaced_model_handles_mixed_dtype_student() -> None:
    dim = 16
    teacher = SimpleTeacherBlock(dim=dim)
    student = MixedStudentBlock(dim=dim)

    # Initial parameter dtypes are mixed
    assert student.self_attn.w1.dtype == torch.float32
    assert student.self_attn.w2.dtype == torch.float16

    calibration_inputs = torch.randn(2, 4, dim, dtype=torch.float32)

    # Micro-healing should run without crashing on mixed dtypes
    stats = _heal_replaced_model_impl(
        teacher_model=teacher,
        student_model=student,
        calibration_inputs=calibration_inputs,
        steps=3,
        also_ridge_init=False,
        device="cpu",
    )

    assert stats["num_healed_layers"] >= 1
    # Check that parameters are restored to their original dtypes
    assert student.self_attn.w1.dtype == torch.float32
    assert student.self_attn.w2.dtype == torch.float16
