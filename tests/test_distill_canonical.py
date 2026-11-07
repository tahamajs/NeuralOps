#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for canonical distillation single source of truth (finding F-92).
"""

from __future__ import annotations

import unittest.mock as mock
import pytest
import torch
import torch.nn as nn

import neuralops.api as napi
import neuralops.trainer.distillation.multi_level_kd as ml_kd


def test_api_distill_uses_trainer_distillation(monkeypatch):
    """Verify api.distill() delegates to canonical trainer.distillation.multi_level_kd.distill."""
    calls = []

    def mock_distill(student, teacher, train_loader, epochs=3, lr=2e-5, **kwargs):
        calls.append({
            "student": student,
            "teacher": teacher,
            "train_loader": train_loader,
            "epochs": epochs,
            "lr": lr,
            "kwargs": kwargs,
        })
        return {"status": "mocked", "loss": 0.123}

    monkeypatch.setattr(ml_kd, "distill", mock_distill)

    student = nn.Linear(10, 2)
    teacher = nn.Linear(10, 2)
    dataset = [torch.randn(4, 10) for _ in range(2)]

    res = napi.distill(
        student=student,
        teacher=teacher,
        dataset=dataset,
        epochs=2,
        lr=1e-4,
        temperature=3.0,
        alpha=0.7,
        custom_flag="test_val",
    )

    assert res == {"status": "mocked", "loss": 0.123}
    assert len(calls) == 1
    assert calls[0]["epochs"] == 2
    assert calls[0]["lr"] == 1e-4
    assert calls[0]["kwargs"].get("temperature") == 3.0
    assert calls[0]["kwargs"].get("alpha") == 0.7
    assert calls[0]["kwargs"].get("custom_flag") == "test_val"
