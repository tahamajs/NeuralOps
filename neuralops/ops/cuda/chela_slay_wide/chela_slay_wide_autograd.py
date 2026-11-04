#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function & Dispatcher for CHELA-SLAY Wide Head Attention CUDA Kernel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.autograd import Function

from neuralops.ops.cuda._bindings import load_cuda_module

logger = logging.getLogger(__name__)

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    try:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "chela_slay_wide_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "chela_slay_wide.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_chela_slay_wide",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["chela_slay_wide_forward"],
        )
        return _MODULE
    except Exception as e:
        logger.debug("Failed to compile or load cuda_chela_slay_wide: %s", e)
        return None


class ChelaSlayWideFunction(Function):
    """Autograd Function wrapping CUDA chela_slay_wide_forward kernel."""

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

        mod = _get_module()
        if mod is not None and hasattr(mod, "chela_slay_wide_forward"):
            return mod.chela_slay_wide_forward(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                int(chunk_size),
            )

        # Level 2 Triton fallback
        try:
            from neuralops.ops.triton.chela_slay_wide_triton import chela_slay_wide_triton

            return chela_slay_wide_triton(q, k, v, chunk_size=chunk_size)
        except Exception as e:
            logger.debug("chela_slay_wide Triton fallback triggered: %s", e)

        # Level 3 PyTorch reference fallback
        from neuralops.ops.triton.chela_slay_wide_triton import _chela_slay_wide_pytorch_ref

        return _chela_slay_wide_pytorch_ref(q, k, v, chunk_size=chunk_size)

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        q, k, v = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        from neuralops.ops.triton.chela_slay_wide_triton import _chela_slay_wide_pytorch_ref

        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            k_req = k.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)
            out_rec = _chela_slay_wide_pytorch_ref(q_req, k_req, v_req, chunk_size=chunk_size)
            dq, dk, dv = torch.autograd.grad(
                outputs=out_rec,
                inputs=[q_req, k_req, v_req],
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        return dq, dk, dv, None


def chela_slay_wide_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    """CHELA-SLAY Wide-Head attention forward operator with 3-tier fallback hierarchy."""
    if not q.is_cuda:
        from neuralops.ops.triton.chela_slay_wide_triton import _chela_slay_wide_pytorch_ref

        return _chela_slay_wide_pytorch_ref(q, k, v, chunk_size=chunk_size, **kwargs)

    return ChelaSlayWideFunction.apply(q, k, v, chunk_size)


__all__ = ["ChelaSlayWideFunction", "chela_slay_wide_attention_forward"]
