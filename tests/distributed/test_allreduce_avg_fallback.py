#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for ReduceOp.AVG fallback in all_reduce."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.distributed as dist

from neuralops.distributed.context import DistributedContext
from neuralops.distributed.tensor_parallel import all_reduce


def test_tensor_parallel_allreduce_avg_fallback_when_avg_missing(monkeypatch):
    """Verify all_reduce falls back to SUM and div when ReduceOp.AVG is missing."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    # Simulate older PyTorch where ReduceOp.AVG does not exist
    reduce_op_mock = MagicMock(spec=[])
    reduce_op_mock.SUM = "SUM"
    monkeypatch.setattr(dist, "ReduceOp", reduce_op_mock)

    recorded_calls = []

    def mock_dist_all_reduce(tensor, op=None, group=None):
        recorded_calls.append((op, group))
        # Simulate summation across 2 ranks: doubles the tensor
        tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", mock_dist_all_reduce)

    x = torch.tensor([10.0, 20.0])
    res = all_reduce(x, op="avg")

    assert len(recorded_calls) == 1
    assert recorded_calls[0][0] == "SUM"
    # Result should be (x * 2) / 2 = x
    assert torch.allclose(res, torch.tensor([10.0, 20.0]))


def test_tensor_parallel_allreduce_avg_fallback_when_avg_fails(monkeypatch):
    """Verify all_reduce falls back to SUM and div when ReduceOp.AVG raises at runtime."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)

    reduce_op_mock = MagicMock()
    reduce_op_mock.AVG = "AVG"
    reduce_op_mock.SUM = "SUM"
    monkeypatch.setattr(dist, "ReduceOp", reduce_op_mock)

    call_count = 0

    def mock_dist_all_reduce(tensor, op=None, group=None):
        nonlocal call_count
        call_count += 1
        if op == "AVG":
            raise RuntimeError("NCCL does not support ReduceOp.AVG on this device")
        elif op == "SUM":
            tensor.mul_(4)

    monkeypatch.setattr(dist, "all_reduce", mock_dist_all_reduce)

    x = torch.tensor([4.0, 8.0])
    res = all_reduce(x, op="avg")

    assert call_count == 2
    assert torch.allclose(res, torch.tensor([4.0, 8.0]))


def test_distributed_context_allreduce_avg_fallback(monkeypatch):
    """Verify DistributedContext.all_reduce falls back to SUM and div when AVG fails."""
    ctx = DistributedContext(init_if_available=False)
    monkeypatch.setattr(ctx, "is_distributed", lambda: True)
    ctx.world_size = 2

    reduce_op_mock = MagicMock()
    reduce_op_mock.AVG = "AVG"
    reduce_op_mock.SUM = "SUM"
    monkeypatch.setattr(dist, "ReduceOp", reduce_op_mock)

    call_count = 0

    def mock_dist_all_reduce(tensor, op=None, group=None):
        nonlocal call_count
        call_count += 1
        if op == "AVG":
            raise RuntimeError("Backend AVG failed")
        elif op == "SUM":
            tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", mock_dist_all_reduce)

    x = torch.tensor([1.0, 3.0])
    res = ctx.all_reduce(x, op="avg")

    assert call_count == 2
    assert torch.allclose(res, torch.tensor([1.0, 3.0]))
