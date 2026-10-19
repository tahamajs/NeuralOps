# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Routing Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "routing_attention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "routing_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_routing_attention", cpp_src, cuda_src, ["routing_attn_fwd", "routing_attn_bwd"])
    return _MODULE


def routing_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    route_weights: torch.Tensor,
    top_k: int = 32,
) -> torch.Tensor:
    B, H, N, D = q.shape
    rw = (
        route_weights.view(B, H, 1, -1)
        if route_weights.numel() == B * H * N
        else (
            route_weights.view(B, H, N, -1).mean(dim=-1, keepdim=True).transpose(-2, -1)
            if route_weights.numel() == B * H * N * 16
            else None
        )
    )
    if rw is not None:
        mask = (rw > 0.0).expand(B, H, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    return torch.nan_to_num(out, nan=0.0)


class _RoutingAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, route_weights, top_k):
        mod = _get_module()
        if mod is not None and hasattr(mod, "routing_attn_fwd") and q.is_cuda and q.shape[-1] <= 128:
            out = mod.routing_attn_fwd(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                route_weights.contiguous(),
                int(top_k),
            )
        else:
            out = routing_attention_ref(q, k, v, route_weights, top_k)

        ctx.save_for_backward(q, k, v, route_weights)
        ctx.top_k = int(top_k)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, route_weights = ctx.saved_tensors
        top_k = ctx.top_k
        mod = _get_module()
        if mod is not None and hasattr(mod, "routing_attn_bwd") and q.is_cuda and q.shape[-1] <= 128:
            grad_q, grad_k, grad_v = mod.routing_attn_bwd(
                grad_output.contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                route_weights.contiguous(),
                top_k,
            )
            return grad_q, grad_k, grad_v, None, None

        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = routing_attention_ref(q_ref, k_ref, v_ref, route_weights, top_k)
        gq, gk, gv = torch.autograd.grad(out, (q_ref, k_ref, v_ref), grad_output)
        return gq, gk, gv, None, None


def routing_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    route_weights: torch.Tensor,
    top_k: int = 32,
) -> torch.Tensor:
    if any(t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)):
        return _RoutingAttentionFunction.apply(q, k, v, route_weights, top_k)

    cuda_fn = None
    if q.shape[-1] <= 128:
        def _cuda_fn(q, k, v, route_weights, top_k=32):
            mod = _get_module()
            return mod.routing_attn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), route_weights.contiguous(), int(top_k))
        cuda_fn = _cuda_fn
    return dispatch(cuda_fn, routing_attention_ref, routing_attention_ref, q, k=k, v=v, route_weights=route_weights, top_k=top_k, kernel_name="routing_attention")


def routing_attn_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    route_weights: torch.Tensor,
    top_k: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mod = _get_module()
    if mod is not None and hasattr(mod, "routing_attn_bwd") and q.is_cuda and q.shape[-1] <= 128:
        return tuple(
            mod.routing_attn_bwd(
                grad_output.contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                route_weights.contiguous(),
                int(top_k),
            )
        )
    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    with torch.enable_grad():
        out = routing_attention_ref(q_ref, k_ref, v_ref, route_weights, top_k)
    return torch.autograd.grad(out, (q_ref, k_ref, v_ref), grad_output)
