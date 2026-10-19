# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Paged KV Cache Management Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
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

    src_idx = block_mapping[:, 0].long()
    dst_idx = block_mapping[:, 1].long()
    valid_mask = (src_idx >= 0) & (src_idx < max_src) & (dst_idx >= 0) & (dst_idx < max_dst)
    valid_src = src_idx[valid_mask]
    valid_dst = dst_idx[valid_mask]

    if valid_dst.numel() > 0:
        dst_k[valid_dst] = src_k[valid_src]
        dst_v[valid_dst] = src_v[valid_src]

    return dst_k, dst_v

copy_blocks_cow_ref = copy_blocks_cow_pytorch_ref
paged_kv_cache_pytorch_ref = copy_blocks_cow_pytorch_ref


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
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor | None, ...]:
        grad_dst_k = grad_outputs[0] if len(grad_outputs) > 0 else None
        grad_dst_v = grad_outputs[1] if len(grad_outputs) > 1 else None
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
                if grad_dst_k is not None:
                    grad_src_k.index_add_(0, src_idx, grad_dst_k[dst_idx])
                if grad_dst_v is not None:
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
