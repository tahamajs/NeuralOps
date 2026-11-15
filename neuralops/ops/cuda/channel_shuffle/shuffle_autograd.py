#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Channel Shuffle Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "shuffle_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "channel_shuffle.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_channel_shuffle",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["channel_shuffle_fwd"],
        )
    return _MODULE


def channel_shuffle_pytorch_ref(
    x: torch.Tensor,
    groups: int = 2,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of channel shuffle.

    Args:
        x: Input tensor of shape (B, C, ...). Must have at least 2 dimensions and
            C divisible by groups.
        groups: Number of groups to divide channels into. Must divide C evenly.

    Returns:
        Tensor with channels shuffled across groups, preserving original shape and dtype.

    Raises:
        ValueError: If x has fewer than 2 dimensions, or groups is non-positive,
            or C is not divisible by groups.
    """
    if x.dim() < 2:
        raise ValueError(
            f"Input tensor x must have at least 2 dimensions (Batch, Channels), got shape {tuple(x.shape)}"
        )
    if groups <= 0:
        raise ValueError(f"groups must be a positive integer, got {groups}")
    if x.shape[1] % groups != 0:
        raise ValueError(f"Number of channels ({x.shape[1]}) must be divisible by groups ({groups})")

    orig_shape = x.shape
    B, C = orig_shape[0], orig_shape[1]
    L = x.numel() // (B * C)
    x_reshaped = x.view(B, groups, C // groups, L)
    x_transposed = x_reshaped.transpose(1, 2).contiguous()
    return x_transposed.view(orig_shape)


def _get_triton_impl():
    try:
        from neuralops.ops.triton.elementwise import channel_shuffle_triton

        return channel_shuffle_triton
    except (ImportError, AttributeError):
        return None


def channel_shuffle(
    x: torch.Tensor,
    groups: int = 2,
) -> torch.Tensor:
    """3-Level Dispatched Channel Shuffle operator.

    Divides channels into groups and interleaves them, supporting Level 1 CUDA C++,
    Level 2 Triton, and Level 3 CPU/fallback execution with AMP float16/bfloat16 compatibility.

    Args:
        x: Input tensor of shape (B, C, ...). Must have at least 2 dimensions and
            C divisible by groups.
        groups: Number of groups to divide channels into. Defaults to 2.

    Returns:
        Tensor with shuffled channels, matching x in shape, dtype, and device.
    """

    def cuda_fn(x, groups=2, **kwargs):
        mod = _get_module()
        return mod.channel_shuffle_fwd(x.contiguous(), int(groups))

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        channel_shuffle_pytorch_ref,
        x,
        groups=groups,
        kernel_name="channel_shuffle",
    )
