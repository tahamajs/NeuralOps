# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Mixture of Experts Routing Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "moe_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "moe_router.cu", "r", encoding="utf-8") as f:
            r_src = f.read()
        with open(curr_dir / "expert_dispatch.cu", "r", encoding="utf-8") as f:
            d_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_moe_routing",
            cpp_sources=cpp_src,
            cuda_sources=r_src + "\n" + d_src,
            functions=["moe_dispatch", "moe_gather", "moe_fused_dispatch"],
        )
    return _MODULE


def _moe_dispatch_pytorch_ref(
    tokens: torch.Tensor,
    topk_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    total_slots: int,
) -> torch.Tensor:
    num_tokens, hidden_dim = tokens.shape
    _top_k = topk_indices.shape[1]
    dispatched = torch.zeros((total_slots, hidden_dim), dtype=tokens.dtype, device=tokens.device)

    valid_mask = (expert_offsets >= 0) & (expert_offsets < total_slots)
    tok_ids = torch.arange(num_tokens, device=tokens.device).unsqueeze(1).expand_as(expert_offsets)
    flat_slots = expert_offsets[valid_mask].long()
    flat_toks = tok_ids[valid_mask]
    dispatched[flat_slots] = tokens[flat_toks]

    return dispatched


def _moe_gather_pytorch_ref(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    num_tokens, top_k = routing_weights.shape
    _hidden_dim = expert_outputs.shape[1]
    total_assignments = expert_outputs.shape[0]

    valid_mask = (expert_offsets >= 0) & (expert_offsets < total_assignments)
    safe_offsets = torch.clamp(expert_offsets, 0, max(0, total_assignments - 1)).long()
    gathered = expert_outputs[safe_offsets] * valid_mask.unsqueeze(-1)
    out = (gathered * routing_weights.unsqueeze(-1)).sum(dim=1).to(expert_outputs.dtype)

    return out


class _MoEDispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        total_slots: int,
    ) -> torch.Tensor:
        mod = _get_module()
        cuda_fn = mod.moe_dispatch if mod is not None else None

        dispatched = dispatch(
            cuda_fn,
            None,
            _moe_dispatch_pytorch_ref,
            tokens,
            topk_indices,
            expert_offsets,
            total_slots,
            kernel_name="moe_dispatch",
        )

        ctx.save_for_backward(expert_offsets)
        ctx.num_tokens = tokens.size(0)
        ctx.hidden_dim = tokens.size(1)
        ctx.tokens_dtype = tokens.dtype
        ctx.tokens_device = tokens.device
        return dispatched

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_dispatched = grad_outputs[0]
        expert_offsets, = ctx.saved_tensors
        total_slots = grad_dispatched.size(0)

        valid_mask = (expert_offsets >= 0) & (expert_offsets < total_slots)
        safe_offsets = torch.clamp(expert_offsets, 0, max(0, total_slots - 1))
        gathered = grad_dispatched[safe_offsets] * valid_mask.unsqueeze(-1)
        grad_tokens = gathered.sum(dim=1).to(ctx.tokens_dtype)

        return grad_tokens, None, None, None


class _MoEGatherAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        expert_outputs: torch.Tensor,
        routing_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        mod = _get_module()
        cuda_fn = mod.moe_gather if mod is not None else None

        out = dispatch(
            cuda_fn,
            None,
            _moe_gather_pytorch_ref,
            expert_outputs,
            routing_weights,
            topk_indices,
            expert_offsets,
            kernel_name="moe_gather",
        )

        ctx.save_for_backward(expert_outputs, routing_weights, expert_offsets)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_out = grad_outputs[0]
        expert_outputs, routing_weights, expert_offsets = ctx.saved_tensors
        hidden_dim = expert_outputs.shape[1]
        total_slots = expert_outputs.shape[0]

        valid_mask = (expert_offsets >= 0) & (expert_offsets < total_slots)
        safe_offsets = torch.clamp(expert_offsets, 0, max(0, total_slots - 1))

        gathered_exp = expert_outputs[safe_offsets]
        grad_routing_weights = (grad_out.unsqueeze(1).float() * gathered_exp.float()).sum(dim=-1).to(routing_weights.dtype) * valid_mask

        weighted_grad = (grad_out.unsqueeze(1) * routing_weights.unsqueeze(-1) * valid_mask.unsqueeze(-1)).view(-1, hidden_dim)
        flat_offsets = safe_offsets.view(-1, 1).expand(-1, hidden_dim)
        grad_expert_outputs = torch.zeros_like(expert_outputs)
        grad_expert_outputs.scatter_add_(0, flat_offsets, weighted_grad)

        return grad_expert_outputs, grad_routing_weights, None, None


def moe_route_pytorch_ref(
    tokens: torch.Tensor,
    gate_logits: torch.Tensor,
    top_k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch vectorized reference for routing weight computation."""
    orig_shape = tokens.shape
    if len(orig_shape) == 3:
        tokens = tokens.view(-1, orig_shape[-1])
        gate_logits = gate_logits.view(-1, gate_logits.shape[-1])

    num_tokens = tokens.shape[0]
    topk_vals, topk_indices = torch.topk(gate_logits, k=top_k, dim=-1)
    routing_weights = F.softmax(topk_vals.float(), dim=-1).to(tokens.dtype)

    expert_offsets = torch.arange(num_tokens * top_k, device=tokens.device, dtype=torch.int32).view(num_tokens, top_k)
    return routing_weights, topk_indices, expert_offsets


def moe_dispatch_gather_pytorch_ref(
    tokens: torch.Tensor,
    gate_logits: torch.Tensor,
    expert_fn: Any,
    top_k: int = 2,
) -> torch.Tensor:
    """End-to-end vectorized PyTorch reference for dispatch + expert execution + gather."""
    orig_shape = tokens.shape
    is_3d = (len(orig_shape) == 3)
    if is_3d:
        B, T, D = orig_shape
        tokens = tokens.view(-1, D)
        gate_logits = gate_logits.view(-1, gate_logits.shape[-1])

    num_tokens, hidden_dim = tokens.shape
    topk_vals, topk_indices = torch.topk(gate_logits, k=top_k, dim=-1)
    routing_weights = F.softmax(topk_vals.float(), dim=-1).to(tokens.dtype)
    expert_offsets = torch.arange(num_tokens * top_k, device=tokens.device, dtype=torch.int32).view(num_tokens, top_k)
    total_slots = num_tokens * top_k

    dispatched = _moe_dispatch_pytorch_ref(tokens, topk_indices, expert_offsets, total_slots)
    expert_outputs = expert_fn(dispatched) if expert_fn is not None else dispatched
    out = _moe_gather_pytorch_ref(expert_outputs, routing_weights, topk_indices, expert_offsets)
    if is_3d:
        out = out.view(B, T, hidden_dim)
    return out


def moe_dispatch_gather(
    tokens: torch.Tensor,
    gate_logits: torch.Tensor,
    expert_fn: Any = None,
    top_k: int = 2,
) -> torch.Tensor:
    """Bitonic MoE Fused Dispatch & Gather with 3-level fallback."""
    orig_shape = tokens.shape
    is_3d = (len(orig_shape) == 3)
    if is_3d:
        B, T, D = orig_shape
        tokens = tokens.view(-1, D)
        gate_logits = gate_logits.view(-1, gate_logits.shape[-1])

    num_tokens, hidden_dim = tokens.shape

    if expert_fn is None:
        expert_fn = lambda x: x

    if torch.is_grad_enabled() and (tokens.requires_grad or gate_logits.requires_grad):
        topk_vals, topk_indices = torch.topk(gate_logits, k=top_k, dim=-1)
        routing_weights = F.softmax(topk_vals.float(), dim=-1).to(tokens.dtype)
        expert_offsets = torch.arange(num_tokens * top_k, device=tokens.device, dtype=torch.int32).view(num_tokens, top_k)
        total_slots = num_tokens * top_k
        dispatched = _MoEDispatchAutograd.apply(tokens, topk_indices, expert_offsets, total_slots)
        expert_outputs = expert_fn(dispatched)
        out = _MoEGatherAutograd.apply(expert_outputs, routing_weights, topk_indices, expert_offsets)
    else:
        mod = _get_module()
        if mod is not None and tokens.is_cuda and gate_logits.is_cuda and hasattr(mod, "moe_fused_dispatch") and top_k <= 32:
            try:
                dispatched, routing_weights, topk_indices = mod.moe_fused_dispatch(tokens, gate_logits, top_k)
                expert_outputs = expert_fn(dispatched)
                empty_offsets = torch.empty(0, device=tokens.device, dtype=torch.int32)
                out = mod.moe_gather(expert_outputs, routing_weights, topk_indices, empty_offsets)
                if is_3d:
                    out = out.view(B, T, hidden_dim)
                return out
            except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
                logger.warning("Fused MoE dispatch failed: %s; falling back", exc, exc_info=True)

        topk_vals, topk_indices = torch.topk(gate_logits, k=top_k, dim=-1)
        routing_weights = F.softmax(topk_vals.float(), dim=-1).to(tokens.dtype)
        expert_offsets = torch.arange(num_tokens * top_k, device=tokens.device, dtype=torch.int32).view(num_tokens, top_k)
        total_slots = num_tokens * top_k

        cuda_dispatch = mod.moe_dispatch if mod is not None else None
        cuda_gather = mod.moe_gather if mod is not None else None

        dispatched = dispatch(
            cuda_dispatch, None, _moe_dispatch_pytorch_ref,
            tokens, topk_indices, expert_offsets, total_slots, kernel_name="moe_dispatch"
        )
        expert_outputs = expert_fn(dispatched)
        out = dispatch(
            cuda_gather, None, _moe_gather_pytorch_ref,
            expert_outputs, routing_weights, topk_indices, expert_offsets, kernel_name="moe_gather"
        )

    if is_3d:
        out = out.view(B, T, hidden_dim)

    return out


def moe_dispatch(
    tokens: torch.Tensor,
    topk_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    total_slots: int,
) -> torch.Tensor:
    """Dispatches tokens to expert slots with 3-level fallback."""
    mod = _get_module()
    cuda_dispatch = mod.moe_dispatch if mod is not None else None
    return dispatch(
        cuda_dispatch, None, _moe_dispatch_pytorch_ref,
        tokens, topk_indices, expert_offsets, total_slots, kernel_name="moe_dispatch"
    )


def moe_gather(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Gathers and weights expert outputs with 3-level fallback."""
    mod = _get_module()
    cuda_gather = mod.moe_gather if mod is not None else None
    return dispatch(
        cuda_gather, None, _moe_gather_pytorch_ref,
        expert_outputs, routing_weights, topk_indices, expert_offsets, kernel_name="moe_gather"
    )


moe_routing = moe_dispatch_gather
moe_routing_ref = moe_dispatch_gather_pytorch_ref
moe_routing_pytorch_ref = moe_dispatch_gather_pytorch_ref
moe_dispatch_gather_ref = moe_dispatch_gather_pytorch_ref

__all__ = [
    "moe_dispatch_gather",
    "moe_dispatch",
    "moe_gather",
    "moe_routing",
    "moe_routing_ref",
    "moe_routing_pytorch_ref",
    "moe_dispatch_gather_ref",
    "moe_dispatch_gather_pytorch_ref",
    "moe_route_pytorch_ref",
    "_moe_dispatch_pytorch_ref",
    "_moe_gather_pytorch_ref",
]
