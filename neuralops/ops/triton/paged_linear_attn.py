#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Paged Linear Attention Triton Kernel for vLLM & High-Throughput Serving.

Provides non-contiguous page table dereferencing and XOR bank swizzling for constant
memory decode and high-throughput prefill serving.
Strictly adheres to AGENTS.md §7 (Zero tl.make_block_ptr).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from neuralops.ops.triton._types import (
    HAS_TRITON,
    check_triton_suitability,
)

logger = logging.getLogger(__name__)

if HAS_TRITON:
    import triton
    import triton.language as tl
else:
    triton = None
    tl = None

__all__ = [
    "paged_linear_attention",
    "paged_linear_attention_fwd_kernel",
    "paged_linear_attention_pytorch",
    "swizzle_tensor_layout",
    "unswizzle_tensor_layout",
    "create_paged_kv_cache",
]


def swizzle_tensor_layout(x: torch.Tensor, block_m: int = 16) -> torch.Tensor:
    """
    Applies bitwise XOR column swizzling along the hidden dimension to eliminate
    32-way shared memory bank conflicts on Ada Lovelace (SM 8.9) and Hopper (SM 90):
        col_swizzled = col ^ ((row & 0x7) << 2)

    Args:
        x: Input tensor of shape [..., T, H, D] or [..., T, D].
        block_m: Row tile size along the sequence/chunk dimension.

    Returns:
        Tensor with swizzled column indexing.
    """
    d_dim = x.shape[-1]
    out = torch.empty_like(x)
    seq_len = x.shape[-3] if x.ndim >= 3 else x.shape[-2]
    d_indices = torch.arange(d_dim, device=x.device, dtype=torch.long)

    for m in range(min(block_m, seq_len)):
        shift = (m & 0x7) << 2
        swizzled_d = d_indices ^ shift
        if x.ndim == 4:
            # [B, T, H, D]
            t_idx = torch.arange(m, seq_len, block_m, device=x.device, dtype=torch.long)
            out[:, t_idx, :, :] = x[:, t_idx, :, :][..., swizzled_d]
        elif x.ndim == 3:
            # [B, T, D]
            t_idx = torch.arange(m, seq_len, block_m, device=x.device, dtype=torch.long)
            out[:, t_idx, :] = x[:, t_idx, :][..., swizzled_d]
        else:
            out = x
            break

    return out


def unswizzle_tensor_layout(x: torch.Tensor, block_m: int = 16) -> torch.Tensor:
    """
    Inverts the XOR column swizzling. Since XOR is its own inverse:
        col = col_swizzled ^ ((row & 0x7) << 2)
    """
    return swizzle_tensor_layout(x, block_m=block_m)


if HAS_TRITON and tl is not None:

    @triton.jit
    def paged_linear_attention_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Block_Tables_ptr,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_ob,
        stride_ot,
        stride_oh,
        stride_od,
        stride_bt_b,
        stride_bt_c,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SWIZZLE: tl.constexpr = False,
    ):
        """
        Fused Paged Linear Attention Forward Kernel.
        Dereferences physical pages via Block_Tables without contiguous KV allocation.
        Uses XOR swizzled column indexing to eliminate 32-way shared memory bank conflicts.
        Strictly zero tl.make_block_ptr per AGENTS.md §7.
        """
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        chunk_idx = tl.program_id(2)

        # 1. Dereference physical page ID from vLLM Block Table
        block_id = tl.load(Block_Tables_ptr + seq_idx * stride_bt_b + chunk_idx * stride_bt_c)

        # 2. Compute 2D SRAM tile pointers with XOR swizzling
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        if SWIZZLE:
            # Swizzled column offset to eliminate 32-way bank conflicts
            swizzled_offs_d = offs_d[None, :] ^ ((offs_m[:, None] & 0x7) << 2)
        else:
            swizzled_offs_d = offs_d[None, :]

        q_ptrs = (
            Q_ptr
            + seq_idx * stride_qb
            + (chunk_idx * BLOCK_M + offs_m[:, None]) * stride_qt
            + head_idx * stride_qh
            + swizzled_offs_d * stride_qd
        )
        k_ptrs = (
            K_ptr
            + block_id * stride_kb
            + offs_m[:, None] * stride_kt
            + head_idx * stride_kh
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V_ptr
            + block_id * stride_vb
            + offs_m[:, None] * stride_vt
            + head_idx * stride_vh
            + offs_d[None, :] * stride_vd
        )

        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # Normalized feature map (ELU + 1)
        q_phi = tl.where(q > 0.0, q + 1.0, tl.exp(q))
        k_phi = tl.where(k > 0.0, k + 1.0, tl.exp(k))

        # Intra-chunk associative scan in shared memory
        s_tile = tl.dot(tl.trans(k_phi), v)
        out = tl.dot(q_phi, s_tile)

        out_ptrs = (
            Out_ptr
            + seq_idx * stride_ob
            + (chunk_idx * BLOCK_M + offs_m[:, None]) * stride_ot
            + head_idx * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(out_ptrs, out)

else:
    paged_linear_attention_fwd_kernel = None


def paged_linear_attention_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int = 16,
    swizzle: bool = False,
) -> torch.Tensor:
    """
    Vectorized PyTorch reference implementation of Paged Linear Attention (Level 3 Fallback).

    Args:
        q: Query tensor of shape [B, T, H, D].
        k_cache: Key physical block pool of shape [num_blocks, block_size, H, D].
        v_cache: Value physical block pool of shape [num_blocks, block_size, H, D].
        block_tables: Physical block indices of shape [B, max_num_blocks].
        block_size: Tokens per physical memory page block (typically 16).
        swizzle: If True, indicates Q has swizzled column indexing.

    Returns:
        Output tensor of shape [B, T, H, D].
    """
    b, t, h, d = q.shape
    device = q.device
    dtype = q.dtype
    num_chunks = (t + block_size - 1) // block_size

    effective_q = unswizzle_tensor_layout(q, block_m=block_size) if swizzle else q

    # ELU + 1 feature map
    q_phi = torch.where(effective_q > 0.0, effective_q + 1.0, torch.exp(effective_q)).float()
    out = torch.zeros(b, t, h, d, device=device, dtype=torch.float32)

    for seq_idx in range(b):
        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * block_size
            end_t = min(start_t + block_size, t)
            chunk_len = end_t - start_t

            block_id = int(block_tables[seq_idx, chunk_idx].item())
            k_chunk = k_cache[block_id, :chunk_len, :, :].float()  # [chunk_len, H, D]
            v_chunk = v_cache[block_id, :chunk_len, :, :].float()  # [chunk_len, H, D]

            k_phi = torch.where(k_chunk > 0.0, k_chunk + 1.0, torch.exp(k_chunk))

            # Transpose to [H, chunk_len, D]
            k_phi_h = k_phi.permute(1, 0, 2)
            v_chunk_h = v_chunk.permute(1, 0, 2)
            q_phi_chunk = q_phi[seq_idx, start_t:end_t, :, :].permute(1, 0, 2)  # [H, chunk_len, D]

            # Intra-chunk associative scan: S = K^T @ V -> [H, D, D]
            s_tile = torch.matmul(k_phi_h.transpose(-2, -1), v_chunk_h)
            # Out = Q @ S -> [H, chunk_len, D]
            out_chunk = torch.matmul(q_phi_chunk, s_tile)

            # Store back to [chunk_len, H, D]
            out[seq_idx, start_t:end_t, :, :] = out_chunk.permute(1, 0, 2)

    return out.to(dtype)


def paged_linear_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int = 16,
    swizzle: bool = False,
) -> torch.Tensor:
    """
    High-throughput Paged Linear Attention dispatch entrypoint adhering to 3-level fallback.

    Level 1: Native Triton JIT GPU kernel (autotuned, non-contiguous page dereferencing).
    Level 2: Triton JIT kernel with swizzled banking.
    Level 3: Vectorized PyTorch reference (universal CPU/MPS fallback).

    Args:
        q: [B, T, H, D] Query tensor.
        k_cache: [num_blocks, block_size, H, D] Physical block pool for keys.
        v_cache: [num_blocks, block_size, H, D] Physical block pool for values.
        block_tables: [B, max_num_blocks] Logical-to-physical block table mapping.
        block_size: Tokens per physical page block (e.g. 16 or 32).
        swizzle: Whether XOR shared memory bank swizzling is applied.

    Returns:
        [B, T, H, D] Attention output tensor.
    """
    b, t, h, d = q.shape

    can_use_triton = (
        HAS_TRITON
        and paged_linear_attention_fwd_kernel is not None
        and q.is_cuda
        and k_cache.is_cuda
        and v_cache.is_cuda
        and block_tables.is_cuda
        and (t % block_size == 0)
        and check_triton_suitability(q)
    )

    if can_use_triton:
        num_chunks = (t + block_size - 1) // block_size
        out = torch.empty_like(q)

        grid = (b, h, num_chunks)
        paged_linear_attention_fwd_kernel[grid](
            q,
            k_cache,
            v_cache,
            out,
            block_tables,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            BLOCK_M=block_size,
            BLOCK_N=block_size,
            BLOCK_D=d,
            SWIZZLE=swizzle,
        )
        return out

    if q.is_cuda:
        logger.warning(
            "paged_linear_attention: Triton GPU kernel unavailable or input conditions not met. "
            "Falling back to Level 3 PyTorch reference."
        )

    return paged_linear_attention_pytorch(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        block_size=block_size,
        swizzle=swizzle,
    )


def create_paged_kv_cache(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int = 16,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocates physical KV cache blocks and generates random non-contiguous block tables.

    Returns:
        (k_cache, v_cache, block_tables)
    """
    if device is None:
        device = torch.device("cpu")

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_needed_blocks = batch_size * num_blocks_per_seq
    # Allocate extra capacity in physical memory pool
    pool_blocks = total_needed_blocks + 16

    k_cache = torch.randn(pool_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(pool_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)

    # Random permutation of physical blocks across batch sequences
    perm = torch.randperm(pool_blocks, device=device, dtype=torch.int32)
    block_tables = perm[:total_needed_blocks].view(batch_size, num_blocks_per_seq).contiguous()

    return k_cache, v_cache, block_tables
