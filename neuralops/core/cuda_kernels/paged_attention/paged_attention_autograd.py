"""
Paged Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
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
        with open(curr_dir / "paged_attention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "paged_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_paged_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["paged_attention_v1"],
        )
    return _MODULE


def paged_attention_pytorch_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int = 16,
    scale: float | None = None,
    max_context_len: int | None = None,
) -> torch.Tensor:
    num_seqs, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    out = torch.zeros_like(q)

    for seq_idx in range(num_seqs):
        ctx_len = int(context_lens[seq_idx].item())
        if ctx_len <= 0:
            continue

        num_blocks = (ctx_len + block_size - 1) // block_size
        k_blocks = []
        v_blocks = []

        for b in range(num_blocks):
            physical_idx = int(block_tables[seq_idx, b].item())
            if physical_idx < 0 or physical_idx >= k_cache.shape[0]:
                continue
            k_block = k_cache[physical_idx].permute(0, 2, 1)  # [num_kv_heads, block_size, head_dim]
            v_block = v_cache[physical_idx].permute(0, 2, 1)
            k_blocks.append(k_block)
            v_blocks.append(v_block)

        if not k_blocks:
            continue

        k_all = torch.cat(k_blocks, dim=1)[:, :ctx_len, :]  # [num_kv_heads, ctx_len, head_dim]
        v_all = torch.cat(v_blocks, dim=1)[:, :ctx_len, :]

        # Repeat KV heads for GQA
        if group_size > 1:
            k_all = k_all.repeat_interleave(group_size, dim=0)
            v_all = v_all.repeat_interleave(group_size, dim=0)

        q_seq = q[seq_idx].unsqueeze(1)  # [num_heads, 1, head_dim]

        # Accumulate softmax in float32 for numerical stability across fp16/bf16
        scores = torch.matmul(q_seq.float(), k_all.float().transpose(-1, -2)) * scale
        probs = F.softmax(scores, dim=-1)
        o_seq = torch.matmul(probs, v_all.float()).squeeze(1).to(q.dtype)
        out[seq_idx] = o_seq

    return out
 
paged_attention_pytorch_ref = paged_attention_pytorch_reference
paged_attention_ref = paged_attention_pytorch_reference


def _paged_attention_triton_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int = 16,
    scale: float | None = None,
    max_context_len: int | None = None,
) -> torch.Tensor:
    """Triton-level fallback implementation for PagedAttention."""
    return paged_attention_pytorch_reference(
        q, k_cache, v_cache, block_tables, context_lens, block_size=block_size, scale=scale
    )


class _PagedAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        mod = _get_module()
        cuda_fn = mod.paged_attention_v1 if mod is not None else None

        out = dispatch(
            cuda_fn,
            _paged_attention_triton_fallback,
            paged_attention_pytorch_reference,
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            block_size,
            scale,
            kernel_name="paged_attention",
        )

        ctx.save_for_backward(q, k_cache, v_cache, block_tables, context_lens)
        ctx.block_size = block_size
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k_cache, v_cache, block_tables, context_lens = ctx.saved_tensors
        block_size = ctx.block_size
        scale = ctx.scale

        grad_q = torch.zeros_like(q)
        grad_k_cache = torch.zeros_like(k_cache)
        grad_v_cache = torch.zeros_like(v_cache)

        num_seqs, num_heads, head_dim = q.shape
        num_kv_heads = k_cache.shape[1]
        group_size = num_heads // num_kv_heads

        for seq_idx in range(num_seqs):
            ctx_len = int(context_lens[seq_idx].item())
            if ctx_len <= 0:
                continue

            num_blocks = (ctx_len + block_size - 1) // block_size
            k_blocks = []
            v_blocks = []
            valid_phys_blocks = []

            for b in range(num_blocks):
                p_idx = int(block_tables[seq_idx, b].item())
                if p_idx < 0 or p_idx >= k_cache.shape[0]:
                    continue
                valid_phys_blocks.append(p_idx)
                k_blocks.append(k_cache[p_idx].permute(0, 2, 1))
                v_blocks.append(v_cache[p_idx].permute(0, 2, 1))

            if not k_blocks:
                continue

            k_all = torch.cat(k_blocks, dim=1)[:, :ctx_len, :]  # [num_kv_heads, ctx_len, head_dim]
            v_all = torch.cat(v_blocks, dim=1)[:, :ctx_len, :]

            k_expanded = k_all.repeat_interleave(group_size, dim=0) if group_size > 1 else k_all
            v_expanded = v_all.repeat_interleave(group_size, dim=0) if group_size > 1 else v_all

            q_seq = q[seq_idx].unsqueeze(1)  # [num_heads, 1, head_dim]
            g_out = grad_out[seq_idx].unsqueeze(1)  # [num_heads, 1, head_dim]

            # Forward attention recomputation
            scores = torch.matmul(q_seq.float(), k_expanded.float().transpose(-1, -2)) * scale
            p = F.softmax(scores, dim=-1)

            # Gradients with respect to V and P
            grad_v_exp = torch.matmul(p.transpose(-1, -2), g_out.float())  # [num_heads, ctx_len, head_dim]
            grad_p = torch.matmul(g_out.float(), v_expanded.float().transpose(-1, -2))  # [num_heads, 1, ctx_len]

            # Softmax backward
            sum_gp_p = (grad_p * p).sum(dim=-1, keepdim=True)
            grad_scores = p * (grad_p - sum_gp_p) * scale

            # Gradients with respect to Q and K
            grad_q_seq = torch.matmul(grad_scores, k_expanded.float()).squeeze(1)  # [num_heads, head_dim]
            grad_k_exp = torch.matmul(grad_scores.transpose(-1, -2), q_seq.float())  # [num_heads, ctx_len, head_dim]

            grad_q[seq_idx] = grad_q_seq.to(q.dtype)

            # Reduce GQA heads
            if group_size > 1:
                grad_k_red = grad_k_exp.view(num_kv_heads, group_size, ctx_len, head_dim).sum(dim=1)
                grad_v_red = grad_v_exp.view(num_kv_heads, group_size, ctx_len, head_dim).sum(dim=1)
            else:
                grad_k_red = grad_k_exp
                grad_v_red = grad_v_exp

            # Scatter token gradients back into physical pages
            for b_idx, p_idx in enumerate(valid_phys_blocks):
                tok_start = b_idx * block_size
                tok_end = min(ctx_len, tok_start + block_size)
                cur_len = tok_end - tok_start
                if cur_len <= 0:
                    continue

                gk_slice = grad_k_red[:, tok_start:tok_end, :].permute(0, 2, 1)  # [num_kv_heads, head_dim, cur_len]
                gv_slice = grad_v_red[:, tok_start:tok_end, :].permute(0, 2, 1)

                grad_k_cache[p_idx, :, :, :cur_len] += gk_slice.to(k_cache.dtype)
                grad_v_cache[p_idx, :, :, :cur_len] += gv_slice.to(v_cache.dtype)

        return grad_q, grad_k_cache, grad_v_cache, None, None, None, None


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int = 16,
    max_context_len: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    if torch.is_grad_enabled() and (q.requires_grad or k_cache.requires_grad or v_cache.requires_grad):
        return _PagedAttentionAutograd.apply(
            q, k_cache, v_cache, block_tables, context_lens, block_size, scale
        )

    mod = _get_module()
    cuda_fn = mod.paged_attention_v1 if mod is not None else None

    return dispatch(
        cuda_fn,
        _paged_attention_triton_fallback,
        paged_attention_pytorch_reference,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size,
        scale,
        kernel_name="paged_attention",
    )
