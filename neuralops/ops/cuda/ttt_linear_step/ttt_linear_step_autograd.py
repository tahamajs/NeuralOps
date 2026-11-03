#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function & Dispatcher for TTT Linear Attention Step CUDA Kernel.
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
        with open(curr_dir / "ttt_linear_step_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "ttt_linear_step.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_ttt_linear_step",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["ttt_linear_step_forward"],
        )
        return _MODULE
    except Exception as e:
        logger.debug("Failed to compile or load cuda_ttt_linear_step: %s", e)
        return None


class TTTLinearStepFunction(Function):
    """Autograd Function wrapping CUDA ttt_linear_step_forward kernel."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        etas: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v, etas)

        mod = _get_module()
        if mod is not None and hasattr(mod, "ttt_linear_step_forward"):
            return mod.ttt_linear_step_forward(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                etas.contiguous(),
            )

        # Level 2 Triton fallback
        try:
            from neuralops.ops.triton.ttt_linear_step_triton import ttt_linear_step_triton

            return ttt_linear_step_triton(q, k, v, etas)
        except Exception as e:
            logger.debug("ttt_linear_step Triton fallback triggered: %s", e)

        # Level 3 PyTorch reference fallback
        from neuralops.ops.triton.ttt_linear_step_triton import _ttt_linear_step_pytorch_ref

        return _ttt_linear_step_pytorch_ref(q, k, v, etas)

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        q, k, v, etas = ctx.saved_tensors

        from neuralops.ops.triton.ttt_linear_step_triton import _ttt_linear_step_pytorch_ref

        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            k_req = k.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)
            etas_req = etas.detach().requires_grad_(True)

            out_rec = _ttt_linear_step_pytorch_ref(q_req, k_req, v_req, etas_req)
            dq, dk, dv, detas = torch.autograd.grad(
                outputs=out_rec,
                inputs=[q_req, k_req, v_req, etas_req],
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        return dq, dk, dv, detas


def ttt_linear_step_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    etas: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """TTT Linear Attention forward operator with 3-tier fallback hierarchy."""
    if not q.is_cuda:
        from neuralops.ops.triton.ttt_linear_step_triton import _ttt_linear_step_pytorch_ref

        return _ttt_linear_step_pytorch_ref(q, k, v, etas, **kwargs)

    return TTTLinearStepFunction.apply(q, k, v, etas)


__all__ = ["TTTLinearStepFunction", "ttt_linear_step_attention_forward"]
