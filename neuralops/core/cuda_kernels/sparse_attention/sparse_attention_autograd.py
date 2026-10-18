"""
Block-Sparse Attention Autograd Interface

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
        with open(curr_dir / "sparse_attention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "sparse_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_sparse_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["block_sparse_attn_fwd", "block_sparse_attn_bwd"],
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
    if block_mask.dim() == 4:
        bm = block_mask
        dense_mask = bm.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)[..., :N, :N].bool()
    elif block_mask.dim() == 2:
        dense_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:N, :N].bool()
        dense_mask = dense_mask.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
    else:
        dense_mask = None
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask, scale=scale)
    return torch.nan_to_num(out, nan=0.0)


def _block_sparse_triton_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 16,
    scale: float | None = None,
) -> torch.Tensor | None:
    try:
        from neuralops.core.triton_kernels.block_sparse import block_sparse_attention
        N = q.shape[2]
        dense_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:N, :N]
        return block_sparse_attention(q, k, v, dense_mask)
    except Exception:
        return None


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
        _block_sparse_triton_wrapper,
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

        mod = _get_module()
        if (
            mod is not None
            and hasattr(mod, "block_sparse_attn_bwd")
            and q.is_cuda
            and block_mask.is_cuda
            and block_mask.dim() == 2
            and q.shape[-1] in (32, 64, 128)
            and block_size in (16, 32, 64)
        ):
            grad_q, grad_k, grad_v = mod.block_sparse_attn_bwd(
                grad_output.contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                block_mask.contiguous(),
                block_size,
                scale,
            )
            return grad_q, grad_k, grad_v, None, None, None

        # Fallback to analytical PyTorch reference backward
        B, H, N, D = q.shape
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        if block_mask.dim() == 4:
            dense_mask = block_mask.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)[..., :N, :N].bool()
        elif block_mask.dim() == 2:
            dense_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:N, :N].bool()
            dense_mask = dense_mask.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
        else:
            dense_mask = torch.ones(B, H, N, N, device=q.device, dtype=torch.bool)
        scores = scores.masked_fill(~dense_mask, float("-inf"))

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


def block_sparse_attn_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 16,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    mod = _get_module()
    if (
        mod is not None
        and hasattr(mod, "block_sparse_attn_bwd")
        and q.is_cuda
        and block_mask.is_cuda
        and block_mask.dim() == 2
        and q.shape[-1] in (32, 64, 128)
        and block_size in (16, 32, 64)
    ):
        return tuple(
            mod.block_sparse_attn_bwd(
                grad_output.contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                block_mask.contiguous(),
                block_size,
                scale,
            )
        )
    # PyTorch reference backward
    B, H, N, D = q.shape
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if block_mask.dim() == 4:
        dense_mask = block_mask.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)[..., :N, :N].bool()
    elif block_mask.dim() == 2:
        dense_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:N, :N].bool()
        dense_mask = dense_mask.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
    else:
        dense_mask = torch.ones(B, H, N, N, device=q.device, dtype=torch.bool)
    scores = scores.masked_fill(~dense_mask, float("-inf"))
    p = F.softmax(scores, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)

    grad_out_f = grad_output.float()
    grad_v = torch.matmul(p.transpose(-1, -2), grad_out_f).to(v.dtype)
    grad_p = torch.matmul(grad_out_f, v.float().transpose(-1, -2))
    sum_p_grad_p = (p * grad_p).sum(dim=-1, keepdim=True)
    grad_scores = p * (grad_p - sum_p_grad_p) * scale
    grad_q = torch.matmul(grad_scores, k.float()).to(q.dtype)
    grad_k = torch.matmul(grad_scores.transpose(-1, -2), q.float()).to(k.dtype)
    return grad_q, grad_k, grad_v
