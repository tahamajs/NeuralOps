"""
Sparse Attention Python Interface & 3-Level Fallback (Module 6).
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
        with open(curr_dir / "sparse_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "block_sparse_attn.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_sparse_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["block_sparse_attn_fwd"],
        )
    return _MODULE


def block_sparse_attn_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 16,
    scale: float | None = None,
) -> torch.Tensor:
    B, H, N, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, H, N, N]

    # Expand block mask [num_q_blocks, num_kv_blocks] to dense [N, N]
    dense_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:N, :N]
    scores = scores.masked_fill(~dense_mask.bool().unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v)


def _block_sparse_forward_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 16,
    scale: float | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    mod = _get_module()
    cuda_fn = mod.block_sparse_attn_fwd if mod is not None else None

    return dispatch(
        cuda_fn,
        None,
        block_sparse_attn_pytorch_ref,
        q,
        k,
        v,
        block_mask,
        block_size,
        scale,
        kernel_name="block_sparse_attn",
    )


class _BlockSparseAttnAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_mask, block_size, scale):
        out = _block_sparse_forward_dispatch(q, k, v, block_mask, block_size, scale)
        ctx.save_for_backward(q, k, v, block_mask)
        ctx.block_size = block_size
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, block_mask = ctx.saved_tensors
        block_size = ctx.block_size
        scale = ctx.scale

        B, H, N, D = q.shape
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        dense_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:N, :N]
        scores = scores.masked_fill(~dense_mask.bool().unsqueeze(0).unsqueeze(0), float("-inf"))

        p = F.softmax(scores, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)

        grad_out_f = grad_output.float()
        grad_v = torch.matmul(p.transpose(-1, -2), grad_out_f).to(v.dtype)

        grad_p = torch.matmul(grad_out_f, v.float().transpose(-1, -2))
        sum_p_grad_p = (p * grad_p).sum(dim=-1, keepdim=True)
        grad_scores = p * (grad_p - sum_p_grad_p) * scale

        grad_q = torch.matmul(grad_scores, k.float()).to(q.dtype)
        grad_k = torch.matmul(grad_scores.transpose(-1, -2), q.float()).to(k.dtype)

        return grad_q, grad_k, grad_v, None, None, None


def block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 16,
    scale: float | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    if any(t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)):
        return _BlockSparseAttnAutograd.apply(q, k, v, block_mask, block_size, scale)

    return _block_sparse_forward_dispatch(q, k, v, block_mask, block_size, scale)

