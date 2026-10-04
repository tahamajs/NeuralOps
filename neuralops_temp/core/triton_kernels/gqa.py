#!/usr/bin/env python3
"""
Triton Grouped Query Attention (GQA) Head Expansion Kernel.

Efficiently repeats KV heads along the head dimension for multi-query and grouped-query attention:
  (k, v) with num_kv_heads -> (k_expanded, v_expanded) with num_heads
"""

from __future__ import annotations

from typing import Any

import torch

from neuralops.core.triton_kernels._common import (
    HAS_TRITON,
    check_triton_suitability,
    gqa_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_gqa_expand_kernel(
        In_ptr,
        Out_ptr,
        stride_in_b,
        stride_in_h,
        stride_in_n,
        stride_in_d,
        stride_out_b,
        stride_out_h,
        stride_out_n,
        stride_out_d,
        B,
        _H_OUT,
        N,
        D,
        NUM_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b_id = tl.program_id(0)
        h_out_id = tl.program_id(1)
        n_id = tl.program_id(2)

        h_in_id = h_out_id // NUM_GROUPS

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        in_ptrs = (
            In_ptr
            + b_id * stride_in_b
            + h_in_id * stride_in_h
            + n_id * stride_in_n
            + offs_d * stride_in_d
        )
        vals = tl.load(in_ptrs, mask=mask_d, other=0.0)

        out_ptrs = (
            Out_ptr
            + b_id * stride_out_b
            + h_out_id * stride_out_h
            + n_id * stride_out_n
            + offs_d * stride_out_d
        )
        tl.store(out_ptrs, vals, mask=mask_d)


def is_supported(
    k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int, **kwargs: Any
) -> bool:
    """Check whether GQA expansion can run on the Triton kernel."""
    suitable, _ = check_triton_suitability(k, "gqa")
    if not suitable or not v.is_cuda:
        return False
    if num_kv_heads <= 0 or num_heads <= 0 or (num_heads % num_kv_heads) != 0:
        return False
    return True


def fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward entrypoint for GQA head expansion: (k_expanded, v_expanded)."""
    if num_kv_heads == num_heads:
        return k, v

    if not is_supported(k, v, num_kv_heads, num_heads, **kwargs):
        return gqa_pytorch_fallback(k, v, num_kv_heads, num_heads, **kwargs)

    B, H_kv, N, D = k.shape
    n_rep = num_heads // num_kv_heads

    k_out = torch.empty((B, num_heads, N, D), dtype=k.dtype, device=k.device)
    v_out = torch.empty((B, num_heads, N, D), dtype=v.dtype, device=v.device)

    block_d = triton.next_power_of_2(D)
    grid = (B, num_heads, N)

    _triton_gqa_expand_kernel[grid](
        k,
        k_out,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        k_out.stride(3),
        B,
        num_heads,
        N,
        D,
        NUM_GROUPS=n_rep,
        BLOCK_D=block_d,
    )
    _triton_gqa_expand_kernel[grid](
        v,
        v_out,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        v_out.stride(3),
        B,
        num_heads,
        N,
        D,
        NUM_GROUPS=n_rep,
        BLOCK_D=block_d,
    )

    return k_out, v_out


def bwd(
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward entrypoint: reduces expanded gradients back to num_kv_heads."""
    if num_kv_heads == num_heads:
        return grad_k, grad_v

    n_rep = num_heads // num_kv_heads
    B, _, N, D = grad_k.shape

    gk = grad_k.view(B, num_kv_heads, n_rep, N, D).sum(dim=2)
    gv = grad_v.view(B, num_kv_heads, n_rep, N, D).sum(dim=2)
    return gk, gv
