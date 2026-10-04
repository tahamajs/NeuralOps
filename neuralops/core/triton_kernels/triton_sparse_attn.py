"""
Triton-Accelerated Block-Sparse Attention.
Directly ingests block routing masks to achieve true sub-quadratic VRAM and FLOPs.
"""

import math
import torch
import torch.nn.functional as F

from typing import Any

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False

    class _DummyTriton:
        @staticmethod
        def jit(*args, **kwargs):
            def decorator(fn):
                return fn
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

        @staticmethod
        def next_power_of_2(n):
            return 1 if n == 0 else 2 ** (int(n) - 1).bit_length()

    class _DummyTL:
        constexpr = Any

    triton = _DummyTriton
    tl = _DummyTL


@triton.jit
def _block_sparse_attn_fwd_kernel(
    Q, K, V, Out,
    BlockMask, TokenMask,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_bmb, stride_bmh, stride_bmq, stride_bmk,
    stride_tmb, stride_tmn,
    Z, H, N_q, N_k,
    num_q_blocks, num_k_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_TOKEN_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    b_idx = off_hz // H
    h_idx = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    o_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    # Load Query block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_q, other=0.0)

    # Initialize running FlashAttention variables
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop ONLY over allowed K blocks
    for start_n in range(0, num_k_blocks):
        # 1. Hardware-level skip: Is this block selected by Coarse-Routing?
        bmask_ptr = BlockMask + b_idx * stride_bmb + h_idx * stride_bmh + start_m * stride_bmq + start_n * stride_bmk
        is_valid_block = tl.load(bmask_ptr)

        if is_valid_block:
            n_offset = start_n * BLOCK_N
            is_future_block = IS_CAUSAL and (n_offset > (start_m * BLOCK_M + BLOCK_M - 1))

            if not is_future_block:
                k_ptrs = K + off_hz * stride_kh + (n_offset + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kd
                v_ptrs = V + off_hz * stride_vh + (n_offset + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vd

                k = tl.load(k_ptrs, mask=(n_offset + offs_n[:, None]) < N_k, other=0.0)
                qk = tl.dot(q, tl.trans(k)) * sm_scale

                # 3. Fine-grained Causal Masking
                if IS_CAUSAL:
                    causal_mask = offs_m[:, None] >= (n_offset + offs_n[None, :])
                    qk = tl.where(causal_mask, qk, float("-inf"))

                # 4. Fine-grained Padding Masking
                if HAS_TOKEN_MASK:
                    tmask_ptr = TokenMask + b_idx * stride_tmb + (n_offset + offs_n[None, :]) * stride_tmn
                    tmask = tl.load(tmask_ptr, mask=(n_offset + offs_n[None, :]) < N_k, other=0)
                    qk = tl.where(tmask == 1, qk, float("-inf"))
                else:
                    bounds_mask = (n_offset + offs_n[None, :]) < N_k
                    qk = tl.where(bounds_mask, qk, float("-inf"))

                # 5. Online Softmax Update
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                p = tl.exp(qk - m_ij[:, None])

                l_ij = tl.sum(p, 1)
                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + l_ij

                acc = acc * alpha[:, None]
                v = tl.load(v_ptrs, mask=(n_offset + offs_n[:, None]) < N_k, other=0.0)
                p = p.to(v.dtype)
                acc += tl.dot(p, v)

                m_i = m_ij

    # Safe division to prevent NaN on fully masked queries
    out_vals = acc / l_i[:, None]
    out_vals = tl.where(m_i[:, None] == float("-inf"), 0.0, out_vals)
    tl.store(o_ptrs, out_vals.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_q)


def triton_block_sparse_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_mask: torch.Tensor, block_size: int,
    token_mask: torch.Tensor | None = None, is_causal: bool = False
) -> torch.Tensor:
    """Wrapper function to invoke the Triton Block-Sparse kernel."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available on this system.")
        
    B, H, N_q, D = q.shape
    _, _, N_k, _ = k.shape
    
    sm_scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    
    # Cast masks to int8 for rapid Triton pointer loading
    block_mask = block_mask.to(torch.int8)
    if token_mask is not None:
        token_mask = token_mask.to(torch.int8)
        stride_tmb = token_mask.stride(0)
        stride_tmn = token_mask.stride(1)
    else:
        stride_tmb = 0
        stride_tmn = 0
        
    grid = (triton.cdiv(N_q, block_size), B * H, 1)
    
    _block_sparse_attn_fwd_kernel[grid](
        q, k, v, out,
        block_mask, token_mask,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
        stride_tmb, stride_tmn,
        B, H, N_q, N_k,
        block_mask.shape[2], block_mask.shape[3],
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        BLOCK_DMODEL=triton.next_power_of_2(D),
        IS_CAUSAL=is_causal,
        HAS_TOKEN_MASK=token_mask is not None,
    )
    return out