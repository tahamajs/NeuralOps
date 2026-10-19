#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Triton Linear Attention Bridge

Backward-compatibility bridge exposing Triton-accelerated causal linear attention kernels.
"""

from __future__ import annotations

from typing import Any
import torch

from neuralops.core.triton_kernels.linear_attn import fwd as _linear_attn_fwd
from neuralops.core.triton_kernels._common import HAS_TRITON


def fast_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    eps: float = 1e-6,
    **kwargs: Any,
) -> torch.Tensor:
    """Canonical wrapper for fast fused linear attention."""
    return _linear_attn_fwd(q, k, v, is_causal=is_causal, eps=eps, **kwargs)


class LinearAttentionTriton:
    @staticmethod
    def apply(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True, eps: float = 1e-6) -> torch.Tensor:
        return fast_linear_attention(q, k, v, is_causal=is_causal, eps=eps)


__all__ = ["fast_linear_attention", "LinearAttentionTriton", "HAS_TRITON"]
