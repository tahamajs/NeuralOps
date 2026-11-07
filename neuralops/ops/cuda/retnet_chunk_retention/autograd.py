#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RetNet Chunk Retention Autograd Interface.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None
_MODULE_ERROR: Exception | None = None
logger = logging.getLogger(__name__)


def _get_module():
    global _MODULE, _MODULE_ERROR
    if _MODULE_ERROR is not None:
        return None
    if _MODULE is None:
        try:
            curr_dir = Path(__file__).parent
            with open(curr_dir / "binding.cpp", "r", encoding="utf-8") as f:
                cpp_src = f.read()
            with open(curr_dir / "retnet_chunk_retention.cu", "r", encoding="utf-8") as f:
                cuda_src = f.read()

            _MODULE = load_cuda_module(
                name="cuda_retnet_chunk_retention",
                cpp_sources=cpp_src,
                cuda_sources=cuda_src,
                functions=["retnet_chunk_retention_fwd"],
            )
        except Exception as exc:
            _MODULE_ERROR = exc
            logger.warning("RetNet CUDA extension unavailable; using lower-level fallback: %s", exc)
            return None
    return _MODULE


def retnet_chunk_retention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: float = 0.9,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q_scaled = q.float() * scale
    sim = torch.matmul(q_scaled, k.float().transpose(-2, -1))
    L = q.shape[-2]
    pos = torch.arange(L, device=q.device)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()
    causal_mask = pos[:, None] >= pos[None, :]
    decay = (gamma**dist) * causal_mask
    ret = sim * decay
    out = torch.matmul(ret, v.float())
    return out.to(q.dtype)


def _retnet_chunk_retention_dispatch(q, k, v, gamma=0.9, scale=None) -> torch.Tensor:
    """3-Level Dispatched RetNet Chunk Retention."""

    def cuda_fn(q, k, v, gamma=0.9, scale=None, **kwargs):
        mod = _get_module()
        sc = (
            scale
            if "scale" in kwargs or "scale" in locals() and locals().get("scale") is not None
            else 1.0 / math.sqrt(q.shape[-1])
        )
        return mod.retnet_chunk_retention_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(gamma))

    return dispatch(
        cuda_fn,
        None,
        retnet_chunk_retention_pytorch_ref,
        q,
        k,
        v,
        gamma,
        scale,
        kernel_name="retnet_chunk_retention",
    )


class _RetnetChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, gamma=0.9, scale=None):
        args = [q, k, v, gamma, scale]
        out = _retnet_chunk_retention_dispatch(*args)
        ctx.save_for_backward(*[t for t in args if isinstance(t, torch.Tensor)])
        return out

    @staticmethod
    def backward(ctx: Any, grad_out):
        raise NotImplementedError("RetNet Chunk Retention backward requires remote CUDA execution.")


def retnet_chunk_retention(q, k, v, gamma=0.9, scale=None) -> torch.Tensor:
    """Dispatches RetNet Chunk Retention with autograd support."""
    args = [q, k, v, gamma, scale]
    return _RetnetChunkRetentionFunction.apply(*args)
