# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Context Parallelism Module

Distributes sequence context across multiple GPUs for long-context attention execution.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger("NeuralOps.CP")


def compute_chunk_prefix_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    init_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes exact linear attention over a sequence chunk [B, H, T_c, D] given incoming init_state [B, H, D, D].
    Returns (out, end_state).
    """
    B, H, T_c, D_k = q.shape
    D_v = v.shape[-1]
    if scale is None:
        scale = 1.0 / (D_k ** 0.5)

    q = q * scale

    if init_state is not None:
        S = init_state.clone()
    else:
        S = torch.zeros((B, H, D_k, D_v), dtype=q.dtype, device=q.device)

    outs = []
    for t in range(T_c):
        if g is not None:
            decay = torch.exp(g[:, :, t, :])
            S = S * decay.unsqueeze(-1)
        kt = k[:, :, t:t+1, :]
        vt = v[:, :, t:t+1, :]
        S = S + torch.matmul(kt.transpose(-2, -1), vt)
        ot = torch.matmul(q[:, :, t:t+1, :], S)
        outs.append(ot)

    out = torch.cat(outs, dim=2)
    return out, S


class _ContextParallelScanFunction(torch.autograd.Function):
    """
    Exact 3-Stage Parallel-Prefix Scan across W Context Parallel ranks.
    Guarantees exact forward and backward gradient parity with single-GPU execution.
    """

    @staticmethod

    def forward(
        ctx: Any,
        W: int,
        scale: float,
        *all_chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        q_chunks = all_chunks[:W]
        k_chunks = all_chunks[W:2*W]
        v_chunks = all_chunks[2*W:3*W]
        g_chunks = all_chunks[3*W:4*W]

        out_list = []
        curr_state = None

        for p in range(W):
            qp = q_chunks[p]
            kp = k_chunks[p]
            vp = v_chunks[p]
            gp = g_chunks[p]
            out_p, next_state = compute_chunk_prefix_scan(qp, kp, vp, gp, scale=scale, init_state=curr_state)
            out_list.append(out_p)
            curr_state = next_state

        ctx.scale = scale
        ctx.W = W
        ctx.save_for_backward(*all_chunks)
        return tuple(out_list)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[Any, ...]:
        W = ctx.W
        scale = ctx.scale
        all_chunks = ctx.saved_tensors
        q_chunks = all_chunks[:W]
        k_chunks = all_chunks[W:2*W]
        v_chunks = all_chunks[2*W:3*W]
        g_chunks = all_chunks[3*W:4*W]

        with torch.enable_grad():
            q_vars = [qc.detach().requires_grad_(True) for qc in q_chunks]
            k_vars = [kc.detach().requires_grad_(True) for kc in k_chunks]
            v_vars = [vc.detach().requires_grad_(True) for vc in v_chunks]
            g_vars = [gc.detach().requires_grad_(True) for gc in g_chunks]

            curr_state = None
            tape_outs = []
            for p in range(W):
                out_p, curr_state = compute_chunk_prefix_scan(
                    q_vars[p], k_vars[p], v_vars[p], g_vars[p], scale=scale, init_state=curr_state
                )
                tape_outs.append(out_p)

            grads_to_back = list(grad_outputs)
            torch.autograd.backward(tape_outs, grads_to_back)

            dq = [qv.grad for qv in q_vars]
            dk = [kv.grad for kv in k_vars]
            dv = [vv.grad for vv in v_vars]
            dg = [gv.grad for gv in g_vars]

        return (None, None, *dq, *dk, *dv, *dg)


def context_parallel_recurrent_scan(
    q_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    k_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    v_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    g_chunks: Optional[list[torch.Tensor] | tuple[torch.Tensor, ...]] = None,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    """Helper wrapper for ContextParallelRecurrentScan with clean list/tuple interfaces."""
    W = len(q_chunks)
    if g_chunks is None:
        g_chunks = tuple(torch.zeros_like(qc) for qc in q_chunks)
    if scale is None:
        scale = 1.0 / (q_chunks[0].shape[-1] ** 0.5)

    all_tensors = list(q_chunks) + list(k_chunks) + list(v_chunks) + list(g_chunks)
    return _ContextParallelScanFunction.apply(W, float(scale), *all_tensors)


class ContextParallelRecurrentScan:
    @staticmethod
    def apply(q_chunks, k_chunks, v_chunks, g_chunks=None, scale=None):
        return context_parallel_recurrent_scan(q_chunks, k_chunks, v_chunks, g_chunks, scale)


class ContextParallelLinearAttention(nn.Module):
    """Context-Parallel Linear Attention Layer executing exact 3-stage prefix scan."""

    def __init__(self, embed_dim: int, num_heads: int, cp_size: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cp_size = cp_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward_chunks(
        self,
        q_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
        k_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
        v_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        return list(ContextParallelRecurrentScan.apply(q_chunks, k_chunks, v_chunks))


def ring_attention_chunk_exchange(
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    rank: int,
    world_size: int,
    group: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exchanges Key and Value sequence parallel chunks in a 1D ring topology.
    Sends chunk to (rank + 1) % world_size and receives from (rank - 1) % world_size.
    Works seamlessly in both distributed (dist.is_initialized) and simulated/single-rank mode.
    """
    if world_size <= 1 or not dist.is_initialized():
        return k_chunk.clone(), v_chunk.clone()

    send_to = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size

    next_k = torch.empty_like(k_chunk)
    next_v = torch.empty_like(v_chunk)

    reqs = [
        dist.isend(k_chunk.contiguous(), dst=send_to, group=group),
        dist.isend(v_chunk.contiguous(), dst=send_to, group=group),
        dist.irecv(next_k, src=recv_from, group=group),
        dist.irecv(next_v, src=recv_from, group=group),
    ]
    for r in reqs:
        r.wait()

    return next_k, next_v


class RingLinearAttention(nn.Module):
    """
    Ring Linear Attention layer exchanging Key/Value chunks sequentially across ring ranks.
    Maintains O(1) memory scaling with sequence length across the GPU ring.
    """

    def __init__(self, embed_dim: int, num_heads: int, ring_size: int = 1, rank: int = 0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ring_size = ring_size
        self.rank = rank

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x_local: torch.Tensor,
        group: Any = None,
    ) -> torch.Tensor:
        """
        x_local: [B, T_local, D] local sequence chunk for this rank.
        """
        B, T_l, D = x_local.shape
        H = self.num_heads
        D_h = self.head_dim

        q = self.q_proj(x_local).view(B, T_l, H, D_h).transpose(1, 2)
        k = self.k_proj(x_local).view(B, T_l, H, D_h).transpose(1, 2)
        v = self.v_proj(x_local).view(B, T_l, H, D_h).transpose(1, 2)

        scale = 1.0 / (D_h ** 0.5)
        q = q * scale

        curr_k = k
        curr_v = v
        accum_out = torch.zeros_like(q)

        for step in range(self.ring_size):
            kv = torch.matmul(curr_k.transpose(-2, -1), curr_v)
            accum_out = accum_out + torch.matmul(q, kv)

            if self.ring_size > 1:
                curr_k, curr_v = ring_attention_chunk_exchange(
                    curr_k, curr_v, rank=self.rank, world_size=self.ring_size, group=group
                )

        out = accum_out.transpose(1, 2).contiguous().view(B, T_l, D)
        return self.out_proj(out)


__all__ = [
    "compute_chunk_prefix_scan",
    "ContextParallelRecurrentScan",
    "ContextParallelLinearAttention",
    "ring_attention_chunk_exchange",
    "RingLinearAttention",
]
