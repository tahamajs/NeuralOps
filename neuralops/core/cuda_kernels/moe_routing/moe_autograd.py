"""
MoE Routing Interface, Autograd Functions, and 3-Level Fallback (Module 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

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
            functions=["moe_dispatch", "moe_gather"],
        )
    return _MODULE


def _moe_dispatch_pytorch_ref(
    tokens: torch.Tensor,
    topk_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    total_slots: int,
) -> torch.Tensor:
    num_tokens, hidden_dim = tokens.shape
    top_k = topk_indices.shape[1]
    dispatched = torch.zeros((total_slots, hidden_dim), dtype=tokens.dtype, device=tokens.device)

    for tok_idx in range(num_tokens):
        for k_idx in range(top_k):
            slot = int(expert_offsets[tok_idx, k_idx].item())
            if 0 <= slot < total_slots:
                dispatched[slot] = tokens[tok_idx]

    return dispatched


def _moe_gather_pytorch_ref(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    num_tokens, top_k = routing_weights.shape
    hidden_dim = expert_outputs.shape[1]
    total_assignments = expert_outputs.shape[0]

    out = torch.zeros((num_tokens, hidden_dim), dtype=expert_outputs.dtype, device=expert_outputs.device)

    for tok_idx in range(num_tokens):
        for k_idx in range(top_k):
            slot = int(expert_offsets[tok_idx, k_idx].item())
            if 0 <= slot < total_assignments:
                w = routing_weights[tok_idx, k_idx]
                out[tok_idx] += w * expert_outputs[slot]

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
    def backward(ctx, grad_dispatched: torch.Tensor):
        expert_offsets, = ctx.saved_tensors
        num_tokens = ctx.num_tokens
        hidden_dim = ctx.hidden_dim
        top_k = expert_offsets.size(1)
        total_slots = grad_dispatched.size(0)

        grad_tokens = torch.zeros((num_tokens, hidden_dim), dtype=ctx.tokens_dtype, device=ctx.tokens_device)

        for tok_idx in range(num_tokens):
            for k_idx in range(top_k):
                slot = int(expert_offsets[tok_idx, k_idx].item())
                if 0 <= slot < total_slots:
                    grad_tokens[tok_idx] += grad_dispatched[slot]

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
    def backward(ctx, grad_out: torch.Tensor):
        expert_outputs, routing_weights, expert_offsets = ctx.saved_tensors
        num_tokens, top_k = routing_weights.shape
        hidden_dim = expert_outputs.shape[1]
        total_slots = expert_outputs.shape[0]

        grad_expert_outputs = torch.zeros_like(expert_outputs)
        grad_routing_weights = torch.zeros_like(routing_weights)

        for tok_idx in range(num_tokens):
            for k_idx in range(top_k):
                slot = int(expert_offsets[tok_idx, k_idx].item())
                if 0 <= slot < total_slots:
                    w = routing_weights[tok_idx, k_idx]
                    grad_expert_outputs[slot] += w * grad_out[tok_idx]
                    # Dot product over hidden dimension
                    grad_routing_weights[tok_idx, k_idx] = (grad_out[tok_idx].float() * expert_outputs[slot].float()).sum().to(routing_weights.dtype)

        return grad_expert_outputs, grad_routing_weights, None, None


def moe_route_pytorch_ref(
    tokens: torch.Tensor,
    gate_logits: torch.Tensor,
    top_k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_shape = tokens.shape
    if len(orig_shape) == 3:
        tokens = tokens.view(-1, orig_shape[-1])
        gate_logits = gate_logits.view(-1, gate_logits.shape[-1])

    num_tokens = tokens.shape[0]
    topk_vals, topk_indices = torch.topk(gate_logits, k=top_k, dim=-1)
    # Softmax in float32 for numerical stability across fp16/bf16
    routing_weights = F.softmax(topk_vals.float(), dim=-1).to(tokens.dtype)

    expert_offsets = torch.arange(num_tokens * top_k, device=tokens.device, dtype=torch.int32).view(num_tokens, top_k)
    return routing_weights, topk_indices, expert_offsets


def moe_dispatch_gather(
    tokens: torch.Tensor,
    gate_logits: torch.Tensor,
    expert_fn: Any,
    top_k: int = 2,
) -> torch.Tensor:
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

    if torch.is_grad_enabled() and (tokens.requires_grad or gate_logits.requires_grad):
        dispatched = _MoEDispatchAutograd.apply(tokens, topk_indices, expert_offsets, total_slots)
        expert_outputs = expert_fn(dispatched)
        out = _MoEGatherAutograd.apply(expert_outputs, routing_weights, topk_indices, expert_offsets)
    else:
        mod = _get_module()
        if mod is not None and tokens.is_cuda:
            try:
                dispatched = mod.moe_dispatch(tokens, topk_indices, expert_offsets, total_slots)
                expert_outputs = expert_fn(dispatched)
                out = mod.moe_gather(expert_outputs, routing_weights, topk_indices, expert_offsets)
                if is_3d:
                    out = out.view(B, T, hidden_dim)
                return out
            except Exception:
                pass

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
