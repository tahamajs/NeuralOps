#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and contract tests for canonical distill() entry point."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neuralops.trainer.distillation import distill


class ToyModel(nn.Module):
    def __init__(self, in_features: int = 16, out_features: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, output_hidden_states: bool = False):
        logits = self.linear(x)
        if output_hidden_states:
            return SimpleNamespace(logits=logits, hidden_states=(logits,))
        return logits


class MockTracker:
    def __init__(self):
        self.logged_metrics: list[tuple[dict[str, Any], int]] = []

    def log_metrics(self, metrics: dict[str, Any], step: int = 0):
        self.logged_metrics.append((metrics, step))


class TestDistillEntryPoint:
    """Verifies distill() API signature, mixed_precision, tracker, and execution."""

    @pytest.fixture
    def dataloader(self):
        x = torch.randn(16, 16)
        ds = TensorDataset(x)
        return DataLoader(ds, batch_size=4)

    def test_distill_canonical_arguments(self, dataloader):
        torch.manual_seed(42)
        student = ToyModel(16, 8)
        teacher = ToyModel(16, 8)
        tracker = MockTracker()

        res = distill(
            student=student,
            teacher=teacher,
            train_loader=dataloader,
            epochs=2,
            lr=1e-3,
            device="cpu",
            temperature=2.0,
            alpha=0.5,
            max_steps=3,
            mixed_precision="no",
            tracker=tracker,
            log_interval=1,
            seed=123,
        )

        assert isinstance(res, dict)
        assert res["status"] == "success"
        assert res["steps_completed"] == 3
        assert res["training_type"] == "kd"
        assert len(tracker.logged_metrics) == 3

    def test_distill_positional_arguments(self, dataloader):
        torch.manual_seed(42)
        student = ToyModel(16, 8)
        teacher = ToyModel(16, 8)

        res = distill(student, teacher, dataloader, epochs=1, max_steps=2)
        assert res["status"] == "success"
        assert res["steps_completed"] == 2

    def test_distill_inverted_kwargs_order(self, dataloader):
        student = ToyModel(16, 8)
        teacher = ToyModel(16, 8)

        res = distill(
            teacher=teacher,
            student=student,
            train_loader=dataloader,
            epochs=1,
            max_steps=2,
        )
        assert res["status"] == "success"

    def test_distill_dataset_kwarg_alias(self):
        student = ToyModel(16, 8)
        teacher = ToyModel(16, 8)
        raw_batches = [torch.randn(4, 16) for _ in range(2)]

        res = distill(
            student=student,
            teacher=teacher,
            dataset=raw_batches,
            epochs=1,
            max_steps=2,
        )
        assert res["status"] == "success"

    def test_distill_bf16_autocast_cpu(self, dataloader):
        student = ToyModel(16, 8)
        teacher = ToyModel(16, 8)

        res = distill(
            student=student,
            teacher=teacher,
            train_loader=dataloader,
            epochs=1,
            max_steps=2,
            mixed_precision="bf16",
        )
        assert res["status"] == "success"

    def test_distill_missing_models_raises(self, dataloader):
        with pytest.raises(ValueError, match="Both 'student' and 'teacher'"):
            distill(student=None, teacher=None, train_loader=dataloader)
