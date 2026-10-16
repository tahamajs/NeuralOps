"""
MoE Routing Interface and 3-Level Fallback (Module 5).
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


def moe_route_pytorch_ref(
    tokens: torch.Tensor,
    gate_logits: torch.Tensor,
    top_k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = gate_logits.shape
    topk_vals, topk_indices = torch.topk(gate_logits, k=top_k, dim=-1)
    routing_weights = F.softmax(topk_vals, dim=-1)

    # Slot assignment
    expert_offsets = torch.arange(num_tokens * top_k, device=tokens.device, dtype=torch.int32).view(num_tokens, top_k)
    return routing_weights, topk_indices, expert_offsets


def moe_dispatch_gather(
    tokens: torch.Tensor,
    gate_logits: torch.Tensor,
    expert_fn: Any,
    top_k: int = 2,
) -> torch.Tensor:
    num_tokens, hidden_dim = tokens.shape
    routing_weights, topk_indices, expert_offsets = moe_route_pytorch_ref(tokens, gate_logits, top_k=top_k)
    total_slots = num_tokens * top_k

    mod = _get_module()
    if mod is not None and tokens.is_cuda:
        dispatched = mod.moe_dispatch(tokens, topk_indices, expert_offsets, total_slots)
        expert_outputs = expert_fn(dispatched)
        out = mod.moe_gather(expert_outputs, routing_weights, topk_indices, expert_offsets)
        return out

    # PyTorch reference
    dispatched = tokens.repeat_interleave(top_k, dim=0)
    expert_outputs = expert_fn(dispatched)
    weighted = expert_outputs.view(num_tokens, top_k, hidden_dim) * routing_weights.unsqueeze(-1)
    return weighted.sum(dim=1)
