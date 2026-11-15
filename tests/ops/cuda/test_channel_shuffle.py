# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for Channel Shuffle CUDA kernel & CPU fallback (Wave 9 / W9-K14).
"""

from __future__ import annotations

import importlib
import inspect
import pytest
import torch
from torch.amp import autocast
import torch.nn.functional as F

import neuralops.ops.cuda as noc


def test_channel_shuffle_cpu_fallback_basic() -> None:
    """Verify channel_shuffle executes on CPU without error and preserves shape and dtype."""
    x = torch.randn(2, 8, 16, 16, dtype=torch.float32)
    out = noc.channel_shuffle(x, groups=2)

    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_channel_shuffle_top_level_dispatch() -> None:
    """Verify noc.channel_shuffle export and basic execution."""
    x = torch.randn(4, 12, 8, dtype=torch.float32)
    out = noc.channel_shuffle(x, groups=3)

    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_channel_shuffle_numerical_parity_vs_torch() -> None:
    """Verify exact numerical parity between channel_shuffle and torch.nn.functional.channel_shuffle."""
    for groups in [2, 4]:
        x = torch.randn(2, 16, 8, 8, dtype=torch.float32)
        out = noc.channel_shuffle(x, groups=groups)
        ref_builtin = F.channel_shuffle(x, groups=groups)
        ref_py = noc.channel_shuffle_pytorch_ref(x, groups=groups)

        torch.testing.assert_close(out, ref_builtin)
        torch.testing.assert_close(out, ref_py)


def test_channel_shuffle_various_dimensions() -> None:
    """Verify channel_shuffle across 2D, 3D, 4D, and 5D tensors."""
    # 2D: (Batch, Channels)
    x2d = torch.randn(4, 12)
    out2d = noc.channel_shuffle(x2d, groups=3)
    ref2d = noc.channel_shuffle_pytorch_ref(x2d, groups=3)
    torch.testing.assert_close(out2d, ref2d)

    # 3D: (Batch, Channels, Length)
    x3d = torch.randn(2, 6, 20)
    out3d = noc.channel_shuffle(x3d, groups=2)
    ref3d = F.channel_shuffle(x3d, 2)
    torch.testing.assert_close(out3d, ref3d)

    # 4D: (Batch, Channels, Height, Width)
    x4d = torch.randn(2, 8, 14, 14)
    out4d = noc.channel_shuffle(x4d, groups=4)
    ref4d = F.channel_shuffle(x4d, 4)
    torch.testing.assert_close(out4d, ref4d)

    # 5D: (Batch, Channels, Depth, Height, Width)
    x5d = torch.randn(1, 4, 3, 5, 5)
    out5d = noc.channel_shuffle(x5d, groups=2)
    ref5d = F.channel_shuffle(x5d, 2)
    torch.testing.assert_close(out5d, ref5d)


def test_channel_shuffle_amp_compliance() -> None:
    """Verify torch.amp compliance (Rule 2.1) and dtype preservation."""
    # Float16
    x_fp16 = torch.randn(2, 8, 10, dtype=torch.float16)
    out_fp16 = noc.channel_shuffle(x_fp16, groups=2)
    assert out_fp16.dtype == torch.float16

    # BFloat16
    x_bf16 = torch.randn(2, 8, 10, dtype=torch.bfloat16)
    out_bf16 = noc.channel_shuffle(x_bf16, groups=2)
    assert out_bf16.dtype == torch.bfloat16

    # Float64
    x_fp64 = torch.randn(2, 8, 10, dtype=torch.float64)
    out_fp64 = noc.channel_shuffle(x_fp64, groups=2)
    assert out_fp64.dtype == torch.float64

    # torch.amp.autocast context
    with autocast("cpu", dtype=torch.bfloat16):
        x = torch.randn(2, 8, 10)
        out = noc.channel_shuffle(x, groups=2)
        assert isinstance(out, torch.Tensor)


def test_channel_shuffle_autograd_backward() -> None:
    """Verify autograd backward gradient propagation through channel shuffle."""
    x = torch.randn(2, 6, 8, requires_grad=True, dtype=torch.float32)
    out = noc.channel_shuffle(x, groups=2)
    loss = (out * 2.5).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    expected_grad = torch.full_like(x, 2.5)
    torch.testing.assert_close(x.grad, expected_grad)


def test_channel_shuffle_validation_errors() -> None:
    """Verify proper validation errors for invalid inputs."""
    # 1D tensor (< 2 dims)
    x_1d = torch.randn(16)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        noc.channel_shuffle(x_1d, groups=2)

    # Non-divisible channels
    x_odd = torch.randn(2, 7, 10)
    with pytest.raises(ValueError, match="divisible by groups"):
        noc.channel_shuffle(x_odd, groups=2)

    # Invalid groups <= 0
    x_valid = torch.randn(2, 8, 10)
    with pytest.raises(ValueError, match="positive integer"):
        noc.channel_shuffle(x_valid, groups=0)


def test_no_forbidden_patterns() -> None:
    """Verify absence of torch.cuda.amp, raw time.time(), etc."""
    shuffle_mod = importlib.import_module("neuralops.ops.cuda.channel_shuffle.shuffle_autograd")
    src = inspect.getsource(shuffle_mod)
    assert "torch.cuda.amp" not in src, "Deprecated torch.cuda.amp found in shuffle_autograd"
    assert "time.time()" not in src, "Unsynchronized time.time() found in shuffle_autograd"
