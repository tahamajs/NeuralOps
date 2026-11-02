#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for fail-closed GPU benchmark device selection."""

import pytest
import torch
from torch import nn

from neuralops.benchmarks.benchmark_all_26_comprehensive import benchmark_single_kernel
from neuralops.benchmarks.benchmark_all_26_kernels import run_all_26_kernels
from scripts.benchmarks.gpu_master_suite import _resolve_requested_device
from scripts.benchmarks.gpu_master_suite import benchmark_single_attention, benchmark_single_compressor


def test_auto_device_resolves_to_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert _resolve_requested_device("auto") == "cuda"


def test_cuda_request_does_not_fallback_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        _resolve_requested_device("cuda:0")


def test_direct_kernel_marks_numerical_mismatch_as_failure():
    value = torch.ones(4)
    result = benchmark_single_kernel(
        "mismatch",
        "test",
        "4",
        lambda tensor: tensor,
        lambda tensor: tensor + 1.0,
        value,
        warmup=0,
        reps=1,
        device=torch.device("cpu"),
    )
    assert result["status"] == "FAIL_NUMERICS"


def test_26_kernel_runner_rejects_cpu_fallback():
    with pytest.raises(RuntimeError, match="GPU-only"):
        run_all_26_kernels(device_str="cpu")


def test_failed_compressor_is_not_replaced_by_svd():
    class BrokenCompressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("constructor failure")

    result = benchmark_single_compressor("broken", BrokenCompressor, device="cpu")
    assert result["status"] == "ERROR"
    assert "constructor failure" in result["error"]
    assert "sqnr_db" not in result


def test_backward_requires_finite_gradients(monkeypatch):
    class DetachedAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, tensor):
            return tensor.detach()

    monkeypatch.setattr(
        "scripts.benchmarks.gpu_master_suite._get_shared_pretrained_attention",
        lambda dim, num_heads: (None, None),
    )
    result = benchmark_single_attention(
        "detached",
        DetachedAttention,
        b=1,
        h=1,
        seq=2,
        d=2,
        device="cpu",
        warmup=0,
        iters=1,
        check_backward=True,
    )
    assert result["backward_passed"] is False
    assert result["status"] == "FAIL_NUMERICAL"
