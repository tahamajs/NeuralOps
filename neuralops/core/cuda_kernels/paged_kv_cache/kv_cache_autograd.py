"""
Paged KV Cache Manager & 3-Level Fallback (Module 8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "kv_cache_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "kv_cache_copy.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_paged_kv_cache",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["copy_blocks_cow"],
        )
    return _MODULE


def copy_blocks_cow_pytorch_ref(
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    block_mapping: torch.Tensor,
    block_size: int = 16,
    num_heads: int = 8,
    head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_mapping.numel() == 0 or block_mapping.size(0) == 0:
        return dst_k, dst_v

    max_src = src_k.shape[0]
    max_dst = dst_k.shape[0]

    for i in range(block_mapping.shape[0]):
        src_idx = int(block_mapping[i, 0].item())
        dst_idx = int(block_mapping[i, 1].item())
        if src_idx < 0 or dst_idx < 0 or src_idx >= max_src or dst_idx >= max_dst:
            continue
        dst_k[dst_idx].copy_(src_k[src_idx])
        dst_v[dst_idx].copy_(src_v[src_idx])

    return dst_k, dst_v


def _copy_blocks_cow_dispatch(
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    block_mapping: torch.Tensor,
    block_size: int = 16,
    num_heads: int = 8,
    head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    mod = _get_module()
    cuda_fn = mod.copy_blocks_cow if mod is not None else None

    return dispatch(
        cuda_fn,
        None,
        copy_blocks_cow_pytorch_ref,
        dst_k,
        dst_v,
        src_k,
        src_v,
        block_mapping,
        block_size,
        num_heads,
        head_dim,
        kernel_name="copy_blocks_cow",
        allow_none=True,
    )


class _CopyBlocksCoWAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dst_k, dst_v, src_k, src_v, block_mapping, block_size, num_heads, head_dim):
        res = _copy_blocks_cow_dispatch(dst_k, dst_v, src_k, src_v, block_mapping, block_size, num_heads, head_dim)
        ctx.save_for_backward(block_mapping)
        ctx.src_shape = src_k.shape
        ctx.dst_shape = dst_k.shape
        ctx.dtype = src_k.dtype
        ctx.device = src_k.device
        if res is not None:
            return res[0], res[1]
        return dst_k, dst_v

    @staticmethod
    def backward(ctx, grad_dst_k, grad_dst_v):
        block_mapping, = ctx.saved_tensors
        grad_src_k = torch.zeros(ctx.src_shape, dtype=ctx.dtype, device=ctx.device)
        grad_src_v = torch.zeros(ctx.src_shape, dtype=ctx.dtype, device=ctx.device)

        if block_mapping.numel() > 0 and block_mapping.size(0) > 0:
            max_src = ctx.src_shape[0]
            max_dst = ctx.dst_shape[0]
            valid = (
                (block_mapping[:, 0] >= 0) & (block_mapping[:, 0] < max_src) &
                (block_mapping[:, 1] >= 0) & (block_mapping[:, 1] < max_dst)
            )
            valid_map = block_mapping[valid]
            if valid_map.numel() > 0:
                src_idx = valid_map[:, 0].long()
                dst_idx = valid_map[:, 1].long()
                grad_src_k.index_add_(0, src_idx, grad_dst_k[dst_idx])
                grad_src_v.index_add_(0, src_idx, grad_dst_v[dst_idx])

        return None, None, grad_src_k, grad_src_v, None, None, None, None


def copy_blocks_cow(
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    block_mapping: torch.Tensor,
    block_size: int = 16,
    num_heads: int = 8,
    head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if any(t.requires_grad for t in (src_k, src_v) if isinstance(t, torch.Tensor)):
        return _CopyBlocksCoWAutograd.apply(
            dst_k, dst_v, src_k, src_v, block_mapping, block_size, num_heads, head_dim
        )

    return _copy_blocks_cow_dispatch(
        dst_k, dst_v, src_k, src_v, block_mapping, block_size, num_heads, head_dim
    )


