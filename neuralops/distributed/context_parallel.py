# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Context-partitioned attention references and a low-level ring exchange primitive."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist


def compute_chunk_prefix_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    init_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a gated, unnormalized causal linear-attention chunk in FP32 state.

    Inputs use ``[B, H, T, D]`` layout. ``g`` contains log-decay values applied
    to the key-state axis before each token update. The return is ``(output,
    end_state)``; the state has shape ``[B, H, D_k, D_v]``. FP16/BF16 inputs
    accumulate in FP32, while FP32/FP64 inputs retain their precision.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, time, features]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError(
            "q and k must have identical shapes; v must match their batch, head, and time axes"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_floating_point():
        raise TypeError("q, k, and v must use a floating-point dtype")
    if g is not None and (g.shape != q.shape or g.device != q.device):
        raise ValueError("g must match q's shape and device")

    B, H, T_c, D_k = q.shape
    D_v = v.shape[-1]
    if B <= 0 or H <= 0 or D_k <= 0 or D_v <= 0:
        raise ValueError("batch, head, and feature dimensions must be positive")
    if scale is None:
        scale = 1.0 / (D_k**0.5)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    if init_state is not None:
        expected_state = (B, H, D_k, D_v)
        if tuple(init_state.shape) != expected_state:
            raise ValueError(
                f"init_state must have shape {expected_state}, got {tuple(init_state.shape)}"
            )
        if init_state.device != q.device:
            raise ValueError("init_state must be on the same device as q")

    output_dtype = q.dtype
    accumulator_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    q_acc = q.to(accumulator_dtype) * scale
    k_acc = k.to(accumulator_dtype)
    v_acc = v.to(accumulator_dtype)
    g_acc = g.to(accumulator_dtype) if g is not None else None

    if init_state is not None:
        S = init_state.to(accumulator_dtype).clone()
    else:
        S = torch.zeros((B, H, D_k, D_v), dtype=accumulator_dtype, device=q.device)

    if T_c == 0:
        return q.new_empty((B, H, 0, D_v)), S

    outs = []
    for t in range(T_c):
        if g_acc is not None:
            decay = torch.exp(g_acc[:, :, t, :])
            S = S * decay.unsqueeze(-1)
        kt = k_acc[:, :, t : t + 1, :]
        vt = v_acc[:, :, t : t + 1, :]
        S = S + torch.matmul(kt.transpose(-2, -1), vt)
        ot = torch.matmul(q_acc[:, :, t : t + 1, :], S)
        outs.append(ot)

    out = torch.cat(outs, dim=2).to(output_dtype)
    return out, S


class _ContextParallelScanFunction(torch.autograd.Function):
    """Differentiable single-process reference over an ordered list of sequence chunks."""

    @staticmethod
    def forward(
        ctx: Any,
        W: int,
        scale: float,
        *all_chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if W <= 0 or len(all_chunks) != 4 * W:
            raise ValueError("the scan requires W > 0 and exactly 4W input tensors")
        q_chunks = all_chunks[:W]
        k_chunks = all_chunks[W : 2 * W]
        v_chunks = all_chunks[2 * W : 3 * W]
        g_chunks = all_chunks[3 * W : 4 * W]

        out_list = []
        curr_state = None

        for p in range(W):
            qp = q_chunks[p]
            kp = k_chunks[p]
            vp = v_chunks[p]
            gp = g_chunks[p]
            out_p, next_state = compute_chunk_prefix_scan(
                qp, kp, vp, gp, scale=scale, init_state=curr_state
            )
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
        k_chunks = all_chunks[W : 2 * W]
        v_chunks = all_chunks[2 * W : 3 * W]
        g_chunks = all_chunks[3 * W : 4 * W]

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

            active = [
                (output, grad)
                for output, grad in zip(tape_outs, grad_outputs)
                if grad is not None
            ]
            variables = [*q_vars, *k_vars, *v_vars, *g_vars]
            if active:
                gradients = torch.autograd.grad(
                    outputs=[item[0] for item in active],
                    inputs=variables,
                    grad_outputs=[item[1] for item in active],
                    allow_unused=True,
                )
            else:
                gradients = (None,) * len(variables)
            gradients = tuple(
                torch.zeros_like(variable) if grad is None else grad
                for variable, grad in zip(variables, gradients)
            )
            dq = gradients[:W]
            dk = gradients[W : 2 * W]
            dv = gradients[2 * W : 3 * W]
            dg = gradients[3 * W : 4 * W]

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
    if W == 0 or len(k_chunks) != W or len(v_chunks) != W:
        raise ValueError("q_chunks, k_chunks, and v_chunks must have the same nonzero length")
    if g_chunks is None:
        g_chunks = tuple(torch.zeros_like(qc) for qc in q_chunks)
    elif len(g_chunks) != W:
        raise ValueError("g_chunks must have the same length as q_chunks")
    reference_shape = q_chunks[0].shape
    if len(reference_shape) != 4:
        raise ValueError("each context chunk must have shape [batch, heads, time, features]")
    if any(chunk.shape != reference_shape for chunk in q_chunks):
        raise ValueError("all q chunks must have matching shapes")
    if any(chunk.shape != reference_shape for chunk in k_chunks):
        raise ValueError("all k chunks must match the q chunk shape")
    if any(chunk.shape[:3] != reference_shape[:3] for chunk in v_chunks):
        raise ValueError("all v chunks must match q batch, head, and time dimensions")
    if any(chunk.shape != reference_shape for chunk in g_chunks):
        raise ValueError("all g chunks must match the q chunk shape")
    if reference_shape[2] == 0:
        raise ValueError("context-parallel reference chunks must contain at least one token")
    reference_device = q_chunks[0].device
    reference_dtype = q_chunks[0].dtype
    all_chunks = (*q_chunks, *k_chunks, *v_chunks, *g_chunks)
    if any(chunk.device != reference_device for chunk in all_chunks):
        raise ValueError("all context chunks must be on the same device")
    if any(chunk.dtype != reference_dtype for chunk in (*q_chunks, *k_chunks, *v_chunks)):
        raise ValueError("q, k, and v chunks must have the same dtype")
    if scale is None:
        scale = 1.0 / (q_chunks[0].shape[-1] ** 0.5)

    all_tensors = list(q_chunks) + list(k_chunks) + list(v_chunks) + list(g_chunks)
    return _ContextParallelScanFunction.apply(W, float(scale), *all_tensors)


class ContextParallelRecurrentScan:
    @staticmethod
    def apply(q_chunks, k_chunks, v_chunks, g_chunks=None, scale=None):
        return context_parallel_recurrent_scan(q_chunks, k_chunks, v_chunks, g_chunks, scale)


class ContextParallelLinearAttention(nn.Module):
    """Single-process reference scan over caller-provided, preprojected context chunks.

    This utility does not distribute tensors or execute collectives. The constructor
    retains the historical dimensional arguments, but chunks passed to
    ``forward_chunks`` must already be projected to ``[B, H, T, D]`` form.
    """

    def __init__(self, embed_dim: int, num_heads: int, cp_size: int = 1) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim and num_heads must be positive and evenly divisible")
        if cp_size <= 0:
            raise ValueError("cp_size must be positive")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cp_size = cp_size

    def forward_chunks(
        self,
        q_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
        k_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
        v_chunks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        if len(q_chunks) != self.cp_size:
            raise ValueError(f"expected {self.cp_size} chunks, got {len(q_chunks)}")
        return list(ContextParallelRecurrentScan.apply(q_chunks, k_chunks, v_chunks))


def ring_attention_chunk_exchange(
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    rank: int,
    world_size: int,
    group: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange K/V chunks in a 1-D ring; this low-level transfer is not autograd-aware."""
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size) and world_size must be positive")
    if k_chunk.shape[:3] != v_chunk.shape[:3] or k_chunk.device != v_chunk.device:
        raise ValueError("K and V chunks must share batch, head, time dimensions and device")
    if world_size == 1:
        return k_chunk.clone(), v_chunk.clone()
    if not dist.is_initialized():
        raise RuntimeError(
            "ring exchange requires initialized torch.distributed when world_size > 1"
        )
    actual_world_size = dist.get_world_size(group=group)
    if actual_world_size != world_size:
        raise ValueError(
            f"world_size={world_size} does not match process-group size {actual_world_size}"
        )
    if dist.get_rank(group=group) != rank:
        raise ValueError(
            "rank must match the current process rank within the supplied process group"
        )

    def global_rank(group_rank: int) -> int:
        if group is None:
            return group_rank
        return dist.get_global_rank(group, group_rank)

    send_to = global_rank((rank + 1) % world_size)
    recv_from = global_rank((rank - 1 + world_size) % world_size)

    send_k = k_chunk.contiguous()
    send_v = v_chunk.contiguous()
    next_k = torch.empty_like(k_chunk)
    next_v = torch.empty_like(v_chunk)

    reqs = [
        dist.isend(send_k, dst=send_to, group=group),
        dist.isend(send_v, dst=send_to, group=group),
        dist.irecv(next_k, src=recv_from, group=group),
        dist.irecv(next_v, src=recv_from, group=group),
    ]
    for r in reqs:
        r.wait()

    return next_k, next_v


class RingLinearAttention(nn.Module):
    """
    Ring Linear Attention layer exchanging Key/Value chunks sequentially across ring ranks.
    Multi-rank execution is inference-only until an autograd-aware ring exchange is
    implemented. The output projection is replicated because each rank emits complete
    embeddings for its own query tokens.
    """

    def __init__(self, embed_dim: int, num_heads: int, ring_size: int = 1, rank: int = 0) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim and num_heads must be positive and evenly divisible")
        if ring_size <= 0 or rank < 0 or rank >= ring_size:
            raise ValueError("rank must be in [0, ring_size) and ring_size must be positive")
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
        if x_local.ndim != 3:
            raise ValueError("x_local must have shape [batch, local_time, embed_dim]")
        B, T_l, D = x_local.shape
        if D != self.embed_dim:
            raise ValueError(f"expected input width {self.embed_dim}, got {D}")
        H = self.num_heads
        D_h = self.head_dim

        q = self.q_proj(x_local).view(B, T_l, H, D_h).transpose(1, 2)
        k = self.k_proj(x_local).view(B, T_l, H, D_h).transpose(1, 2)
        v = self.v_proj(x_local).view(B, T_l, H, D_h).transpose(1, 2)

        scale = 1.0 / (D_h**0.5)

        if self.ring_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "RingLinearAttention requires initialized torch.distributed "
                    "for ring_size > 1"
                )
            if dist.get_world_size(group=group) != self.ring_size:
                raise ValueError("ring_size does not match the process-group size")
            has_trainable_parameters = any(
                parameter.requires_grad for parameter in self.parameters()
            )
            if torch.is_grad_enabled() and has_trainable_parameters:
                raise RuntimeError(
                    "multi-rank RingLinearAttention currently supports inference only; "
                    "P2P chunk exchange does not propagate remote-rank gradients"
                )

        curr_k = k
        curr_v = v
        accumulator_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        q_acc = q.to(accumulator_dtype) * scale
        accum_out = torch.zeros_like(q_acc)

        for step in range(self.ring_size):
            kv = torch.matmul(
                curr_k.to(accumulator_dtype).transpose(-2, -1),
                curr_v.to(accumulator_dtype),
            )
            accum_out = accum_out + torch.matmul(q_acc, kv)

            if self.ring_size > 1:
                curr_k, curr_v = ring_attention_chunk_exchange(
                    curr_k, curr_v, rank=self.rank, world_size=self.ring_size, group=group
                )

        out = accum_out.transpose(1, 2).contiguous().view(B, T_l, D).to(self.out_proj.weight.dtype)
        return self.out_proj(out)


__all__ = [
    "compute_chunk_prefix_scan",
    "ContextParallelRecurrentScan",
    "ContextParallelLinearAttention",
    "ring_attention_chunk_exchange",
    "RingLinearAttention",
]
