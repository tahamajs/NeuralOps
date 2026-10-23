#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Samba MoE Combine Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "samba_moe_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "samba_moe.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_samba_moe", cpp_src, cuda_src, ["samba_moe_combine_fwd"])
    return _MODULE


def samba_moe_ref(
    expert_out1: torch.Tensor,
    expert_out2: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    w1 = routing_weights[..., 0:1]
    w2 = routing_weights[..., 1:2]
    return w1 * expert_out1 + w2 * expert_out2


def _fwd_dispatch(expert_out1, expert_out2, routing_weights):
    def cuda_fn(expert_out1, expert_out2, routing_weights):
        mod = _get_module()
        # The CUDA kernel dispatches on the expert dtype and reads routing
        # weights as that same scalar type.  Router softmax commonly emits
        # FP32 while expert activations are FP16/BF16, so cast explicitly.
        weights = routing_weights.to(dtype=expert_out1.dtype)
        return mod.samba_moe_combine_fwd(
            expert_out1.contiguous(), expert_out2.contiguous(), weights.contiguous()
        )

    return dispatch(
        cuda_fn, samba_moe_ref, samba_moe_ref, expert_out1, expert_out2, routing_weights, kernel_name="samba_moe"
    )


class _SambaMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        expert_out1: torch.Tensor,
        expert_out2: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        out = _fwd_dispatch(expert_out1, expert_out2, routing_weights)
        ctx.save_for_backward(expert_out1, expert_out2, routing_weights)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0]
        expert_out1, expert_out2, routing_weights = ctx.saved_tensors

        w1 = routing_weights[..., 0:1].to(dout.dtype)
        w2 = routing_weights[..., 1:2].to(dout.dtype)

        d_out1 = None
        if ctx.needs_input_grad[0]:
            d_out1 = dout * w1

        d_out2 = None
        if ctx.needs_input_grad[1]:
            d_out2 = dout * w2

        d_weights = None
        if ctx.needs_input_grad[2]:
            dw1 = (dout.float() * expert_out1.float()).sum(dim=-1, keepdim=True)
            dw2 = (dout.float() * expert_out2.float()).sum(dim=-1, keepdim=True)
            d_weights = torch.cat([dw1, dw2], dim=-1).to(routing_weights.dtype)

        return d_out1, d_out2, d_weights


def samba_moe(
    expert_out1: torch.Tensor,
    expert_out2: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Samba MoE Combine with full autograd support."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (expert_out1, expert_out2, routing_weights) if isinstance(t, torch.Tensor)
    )

    if needs_grad:
        return _SambaMoEFunction.apply(expert_out1, expert_out2, routing_weights)

    return _fwd_dispatch(expert_out1, expert_out2, routing_weights)

samba_moe_combine = samba_moe
samba_moe_combine_pytorch_ref = samba_moe_ref
