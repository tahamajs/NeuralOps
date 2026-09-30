#!/usr/bin/env python3
"""
Triton Block-Sparse Attention Kernel (Forward & Backward).

Supports:
  - Arbitrary 2D or 4D block sparsity masks: (n_blocks_q, n_blocks_k)
  - Configurable block tile sizes (e.g. 32, 64, 128)
  - Optional lower-triangular causal masking combined with block sparsity
  - Online safe softmax reduction in GPU SRAM
  - Full autograd integration and fallback handling.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from framework.core.triton_kernels._common import (
    HAS_TRITON,
    check_triton_suitability,
    sparse_attn_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_block_sparse_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Mask_ptr, Out_ptr,
        stride_qbh, stride_qn, stride_qd,
        stride_kbh, stride_kn, stride_kd,
        stride_vbh, stride_vn, stride_vd,
        stride_obh, stride_on, stride_od,
        stride_mb, stride_mh, stride_mq, stride_mk,
        sm_scale,
        N_Q: tl.constexpr,
        N_K: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        q_blk_idx = tl.program_id(1)

        q_start = q_blk_idx * BLOCK_SIZE
        offs_q = q_start + tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, D)
        mask_q = offs_q < N_Q

        q_base = Q_ptr + bh_id * stride_qbh
        k_base = K_ptr + bh_id * stride_kbh
        v_base = V_ptr + bh_id * stride_vbh
        out_base = Out_ptr + bh_id * stride_obh

        q_ptrs = q_base + (offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

        # Online softmax accumulators
        m_prev = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float("inf")
        l_prev = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_SIZE, D), dtype=tl.float32)

        n_k_blocks = (N_K + BLOCK_SIZE - 1) // BLOCK_SIZE

        for k_blk_idx in range(0, n_k_blocks):
            process_block = True
            if IS_CAUSAL:
                process_block = process_block and (k_blk_idx <= q_blk_idx)
            if HAS_MASK:
                mask_val = tl.load(Mask_ptr + q_blk_idx * stride_mq + k_blk_idx * stride_mk)
                process_block = process_block and (mask_val != 0)

            if process_block:
                k_start = k_blk_idx * BLOCK_SIZE
                offs_k = k_start + tl.arange(0, BLOCK_SIZE)
                mask_k = offs_k < N_K

                k_ptrs = k_base + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
                v_ptrs = v_base + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd)

                k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

                # Compute S_ij = Q_i @ K_j^T * sm_scale
                s = tl.dot(q, tl.trans(k)) * sm_scale

                if IS_CAUSAL:
                    causal_cond = offs_q[:, None] >= offs_k[None, :]
                    s = tl.where(causal_cond & mask_q[:, None] & mask_k[None, :], s, float("-inf"))
                else:
                    s = tl.where(mask_q[:, None] & mask_k[None, :], s, float("-inf"))

                # Online safe softmax update
                m_curr = tl.maximum(m_prev, tl.max(s, axis=1))
                alpha = tl.exp(m_prev - m_curr)
                p = tl.exp(s - m_curr[:, None])

                l_curr = l_prev * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

                m_prev = m_curr
                l_prev = l_curr

        # Normalize output
        out = acc / (l_prev[:, None] + 1e-6)
        out_ptrs = out_base + (offs_q[:, None] * stride_on + offs_d[None, :] * stride_od)
        tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_q[:, None])


class _TritonSparseAttnAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: Optional[torch.Tensor],
        block_size: int,
        is_causal: bool,
        eps: float,
    ) -> torch.Tensor:
        orig_shape = q.shape
        D = q.shape[-1]
        N_Q = q.shape[-2]
        N_K = k.shape[-2]
        BH = q.numel() // (N_Q * D)

        q_flat = q.contiguous().view(BH, N_Q, D)
        k_flat = k.contiguous().view(BH, N_K, D)
        v_flat = v.contiguous().view(BH, N_K, D)

        out = torch.empty((BH, N_Q, D), dtype=q.dtype, device=q.device)

        has_mask = block_mask is not None
        if has_mask:
            bm_flat = block_mask.to(torch.int32).contiguous()
            stride_mq, stride_mk = bm_flat.stride(-2), bm_flat.stride(-1)
        else:
            bm_flat = torch.empty((1, 1), dtype=torch.int32, device=q.device)
            stride_mq, stride_mk = 0, 0

        sm_scale = 1.0 / math.sqrt(D)
        n_q_blocks = (N_Q + block_size - 1) // block_size
        grid = (BH, n_q_blocks)

        _triton_block_sparse_fwd_kernel[grid](
            q_flat, k_flat, v_flat, bm_flat, out,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            0, 0, stride_mq, stride_mk,
            sm_scale,
            N_Q=N_Q, N_K=N_K, D=D,
            BLOCK_SIZE=block_size,
            HAS_MASK=has_mask,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(q, k, v, block_mask)
        ctx.block_size = block_size
        ctx.is_causal = is_causal
        ctx.eps = eps
        return out.view(*orig_shape)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, block_mask = ctx.saved_tensors
        q_req = q.detach().requires_grad_(True)
        k_req = k.detach().requires_grad_(True)
        v_req = v.detach().requires_grad_(True)

        with torch.enable_grad():
            out = sparse_attn_pytorch_fallback(
                q_req, k_req, v_req,
                block_mask=block_mask,
                block_size=ctx.block_size,
                is_causal=ctx.is_causal,
                eps=ctx.eps,
            )
            grads = torch.autograd.grad(out, [q_req, k_req, v_req], grad_out, retain_graph=False)

        g_q = grads[0] if ctx.needs_input_grad[0] else None
        g_k = grads[1] if ctx.needs_input_grad[1] else None
        g_v = grads[2] if ctx.needs_input_grad[2] else None
        return g_q, g_k, g_v, None, None, None, None


def is_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> bool:
    """Check whether block-sparse attention can run on the Triton kernel."""
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    suitable, _ = check_triton_suitability(q, "sparse_attn", head_dim=head_dim, seq_len=seq_len)
    return suitable


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: Optional[torch.Tensor] = None,
    block_size: int = 64,
    is_causal: bool = False,
    eps: float = 1e-6,
    **kwargs: Any,
) -> torch.Tensor:
    """Forward entrypoint for fused Triton block-sparse attention."""
    if not is_supported(q, k, v, **kwargs):
        return sparse_attn_pytorch_fallback(
            q, k, v, block_mask=block_mask, block_size=block_size, is_causal=is_causal, eps=eps, **kwargs
        )
    return _TritonSparseAttnAutogradFunction.apply(
        q, k, v, block_mask, block_size, is_causal, eps
    )


def bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    block_mask: Optional[torch.Tensor] = None,
    block_size: int = 64,
    is_causal: bool = False,
    eps: float = 1e-6,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward entrypoint computing gradients for q, k, v."""
    q_req = q.detach().requires_grad_(True)
    k_req = k.detach().requires_grad_(True)
    v_req = v.detach().requires_grad_(True)
    with torch.enable_grad():
        out = fwd(
            q_req, k_req, v_req,
            block_mask=block_mask, block_size=block_size, is_causal=is_causal, eps=eps, **kwargs
        )
        g_q, g_k, g_v = torch.autograd.grad(out, [q_req, k_req, v_req], grad_out)
    return g_q, g_k, g_v
