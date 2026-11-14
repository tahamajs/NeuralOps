#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for distributed context failure when single device fallback is disabled."""

from __future__ import annotations

from unittest.mock import patch
import pytest
import torch.distributed as dist

from neuralops.distributed.context import DistributedContext
from neuralops.distributed.errors import DistributedError


def test_context_fails_when_init_process_group_raises(monkeypatch):
    """Verify DistributedError is raised when distributed init fails and allow_single_device_fallback is False."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def mock_init(*args, **kwargs):
        raise RuntimeError("Process group network initialization failed")

    with patch.object(dist, "init_process_group", side_effect=mock_init):
        with pytest.raises(DistributedError, match="Failed to initialize distributed process group"):
            DistributedContext(backend="gloo", init_if_available=True, allow_single_device_fallback=False)


def test_context_fallback_allowed_when_flag_true(monkeypatch, caplog):
    """Verify context falls back to single-device mode when allow_single_device_fallback is True."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def mock_init(*args, **kwargs):
        raise RuntimeError("Process group network initialization failed")

    with patch.object(dist, "init_process_group", side_effect=mock_init):
        ctx = DistributedContext(backend="gloo", init_if_available=True, allow_single_device_fallback=True)
        assert ctx.world_size == 1
        assert ctx.rank == 0
        assert not ctx.is_distributed()
        assert "Falling back to single-device mode" in caplog.text
