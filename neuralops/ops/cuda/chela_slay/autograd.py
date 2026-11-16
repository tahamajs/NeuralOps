#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function & Dispatcher for CHELA-SLAY Attention CUDA Kernel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.autograd import Function

from neuralops.ops.cuda._bindings import load_cuda_module

logger = logging.getLogger(__name__)

_OPS: Any | None = None


def _get_chela_slay_ops() -> Any | None:
    """Retrieve or JIT-compile the chela_slay CUDA module via content-addressed caching."""
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import chela_slay_ops

        _OPS = chela_slay_ops
        return _OPS
    except ImportError:
        pass

    try:
        pkg_dir = Path(__file__).parent
        with open(pkg_dir / "binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(pkg_dir / "chela_slay.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _OPS = load_cuda_module(
            name="cuda_chela_slay",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["chela_slay_attention_forward"],
            extra_cuda_cflags=["-O3"],
        )
        return _OPS
    except Exception as exc:
        logger.debug("Failed to compile or load cuda_chela_slay: %s", exc)
        return None


class ChelaSlayAttentionFunction(Function):
    """Autograd Function wrapping CUDA chela_slay_attention_forward kernel with full CPU/reference fallback."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)
        ctx.chunk_size = chunk_size

        if q.is_cuda and k.is_cuda and v.is_cuda:
            ops = _get_chela_slay_ops()
            if ops is not None and hasattr(ops, "chela_slay_attention_forward"):
                return ops.chela_slay_attention_forward(
                    q.contiguous().float(),
                    k.contiguous().float(),
                    v.contiguous().float(),
                    chunk_size,
                )

            # Triton fallback if available on CUDA
            try:
                from neuralops.ops.triton.chela_slay_triton import chela_slay_forward

                return chela_slay_forward(q, k, v, chunk_size=chunk_size)
            except Exception as exc:
                logger.debug("CHELA-SLAY Triton fallback failed: %s", exc)

        # PyTorch reference fallback (CPU or CUDA fallback)
        from neuralops.ops.triton.chela_slay_triton import _chela_slay_pytorch

        return _chela_slay_pytorch(q, k, v, chunk_size=chunk_size)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        q, k, v = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        if not (q.requires_grad or k.requires_grad or v.requires_grad):
            return None, None, None, None

        from neuralops.ops.triton.chela_slay_triton import _chela_slay_pytorch

        with torch.enable_grad():
            q_req = q.detach().requires_grad_(q.requires_grad)
            k_req = k.detach().requires_grad_(k.requires_grad)
            v_req = v.detach().requires_grad_(v.requires_grad)
            out_rec = _chela_slay_pytorch(q_req, k_req, v_req, chunk_size=chunk_size)
            dq, dk, dv = torch.autograd.grad(
                outputs=out_rec,
                inputs=[q_req, k_req, v_req],
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        return dq, dk, dv, None


def chela_slay_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    """CHELA-SLAY attention forward operator with CUDA kernel and PyTorch/Triton fallback."""
    if not q.is_cuda:
        from neuralops.ops.triton.chela_slay_triton import _chela_slay_pytorch

        return _chela_slay_pytorch(q, k, v, chunk_size=chunk_size, **kwargs)

    return ChelaSlayAttentionFunction.apply(q, k, v, chunk_size)


__all__ = ["ChelaSlayAttentionFunction", "chela_slay_attention_forward"]
