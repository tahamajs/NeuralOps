#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HSTU Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE: Any = None


def _get_module() -> Any:
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "hstu_attention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hstu_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_hstu_attention", cpp_src, cuda_src, ["hstu_fwd"])
    return _MODULE


def hstu_attention_ref(
    u: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch reference implementation with FP32 accumulator precision.

    Args:
        u: Input tensor of shape (B, N, D).
        v: Input value tensor of shape (B, N, D).
        decay: Decay factor tensor of shape (B, N, D).
        eps: Small epsilon for numerical stability in normalization.

    Returns:
        Output tensor of shape (B, N, D) matching u.dtype and u.device.
    """
    B, N, D = u.shape
    out = torch.empty_like(u)
    s = torch.zeros(B, D, device=u.device, dtype=torch.float32)
    u_f = u.float()
    v_f = v.float()
    decay_f = decay.float()
    for t in range(N):
        s = decay_f[:, t, :] * s + v_f[:, t, :]
        norm = s.abs() + eps
        out[:, t, :] = (u_f[:, t, :] * (s / norm)).to(u.dtype)
    return out


def hstu_attention(
    u: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """HSTU (Hierarchical Sequential Transduction Unit) attention operator.

    Args:
        u: Input tensor of shape (B, N, D).
        v: Value tensor of shape (B, N, D).
        decay: Decay factor tensor of shape (B, N, D).
        eps: Epsilon for normalization stability.

    Returns:
        Output tensor of shape (B, N, D).
    """
    def cuda_fn(
        u: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> torch.Tensor:
        mod = _get_module()
        return mod.hstu_fwd(u.contiguous(), v.contiguous(), decay.contiguous(), float(eps))

    return dispatch(
        cuda_fn,
        hstu_attention_ref,
        hstu_attention_ref,
        u,
        v,
        decay,
        eps=eps,
        kernel_name="hstu_attention",
    )


__all__ = ["hstu_attention", "hstu_attention_ref"]
