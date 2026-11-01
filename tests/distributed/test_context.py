#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for DistributedContext foundation and error handling.
"""

from __future__ import annotations

import pytest
import torch

from neuralops.distributed.context import DistributedContext
from neuralops.distributed.errors import (
    DistributedError,
    E_DIST_INIT_FAILED,
    E_DIST_NOT_INITIALIZED,
    E_DIST_COMM_FAILED,
)


def test_context_detects_1_gpu_falls_back_to_single():
    """Verify context initialization on CPU or single GPU falls back to single-device."""
    ctx = DistributedContext(init_if_available=False)
    assert ctx.world_size == 1
    assert ctx.rank == 0
    assert ctx.local_rank == 0
    assert not ctx.is_distributed()
    assert ctx.is_main_process()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_context_detects_2_gpus():
    """Verify context discovers 2 GPUs when present on the host."""
    ctx = DistributedContext(init_if_available=False)
    assert ctx.available_gpus >= 2


def test_get_device_returns_cuda0_on_single_gpu():
    """Verify get_device returns cuda:0 when GPU is present or cpu on CPU-only hosts."""
    ctx = DistributedContext(init_if_available=False)
    device = ctx.get_device()
    if torch.cuda.is_available() and ctx.available_gpus > 0:
        assert device == torch.device("cuda:0")
    else:
        assert device == torch.device("cpu")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_get_device_returns_cuda1_for_rank_1():
    """Verify get_device(rank=1) maps to cuda:1 when 2+ GPUs exist."""
    ctx = DistributedContext(init_if_available=False)
    device = ctx.get_device(rank=1)
    assert device == torch.device("cuda:1")


def test_barrier_noop_when_not_distributed():
    """Verify barrier does not block or raise when not running in distributed mode."""
    ctx = DistributedContext(init_if_available=False)
    assert not ctx.is_distributed()
    ctx.barrier()


def test_all_reduce_sum_correct():
    """Verify all_reduce behaves as identity in single-process mode or sums in distributed."""
    ctx = DistributedContext(init_if_available=False)
    x = torch.tensor([1.0, 2.0, 3.0])
    res = ctx.all_reduce(x, op="sum")
    if not ctx.is_distributed():
        assert torch.equal(res, torch.tensor([1.0, 2.0, 3.0]))
    else:
        assert res.sum() > 0


def test_broadcast_rank_0_to_all():
    """Verify broadcast returns unchanged tensor in single-process mode."""
    ctx = DistributedContext(init_if_available=False)
    x = torch.tensor([42.0])
    res = ctx.broadcast(x, src=0)
    assert torch.equal(res, torch.tensor([42.0]))


def test_distributed_errors():
    """Verify DistributedError hierarchy and error code attributes."""
    err = DistributedError("init failure", code=E_DIST_INIT_FAILED)
    assert err.code == E_DIST_INIT_FAILED
    assert "E_DIST_INIT_FAILED" in str(err)
    assert issubclass(DistributedError, Exception)

    err_comm = DistributedError("comm timeout", code=E_DIST_COMM_FAILED)
    assert err_comm.code == E_DIST_COMM_FAILED

    err_uninit = DistributedError("not initialized", code=E_DIST_NOT_INITIALIZED)
    assert err_uninit.code == E_DIST_NOT_INITIALIZED
