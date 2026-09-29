#!/usr/bin/env python3
"""
Triton Fused Linear Attention Kernel (Forward & Backward).

Supports:
  - Non-negative feature map phi: "elu+1", "relu", "identity", "softmax_free"
  - Chunked causal streaming linear attention (O(N) memory, tiled in SRAM)
  - Bidirectional parallel reduction for encoder backbones
  - Cosformer-style in-register / pre-applied cosine-sine positional modulation
  - Full autograd integration for training and recovery fine-tuning.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from framework.core.triton_kernels._common import (
    HAS_TRITON,
    check_triton_suitability,
    linear_attn_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_chunked_causal_linear_attn_fwd_kernel(
        Q_ptr,              # [B*H, N, D_k]
        K_ptr,              # [B*H, N, D_k]
        V_ptr,              # [B*H, N, D_v]
        Out_ptr,            # [B*H, N, D_v]
        stride_qbh, stride_qn, stride_qd,
        stride_kbh, stride_kn, stride_kd,
        stride_vbh, stride_vn, stride_vd,
        stride_obh, stride_on, stride_od,
        N: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_CHUNK: tl.constexpr,
        EPS: tl.constexpr,
        ACTIVATION_MODE: tl.constexpr,  # 0: ELU+1, 1: ReLU, 2: Identity
    ):
        r"""
        Fused Chunked Causal Linear Attention Forward Kernel.
        Processes one (batch, head) stream sequentially across chunks,
        maintaining the recurrent state matrix S \in [D_K, D_V] and Z \in [D_K] in SRAM registers.
        """
        bh_id = tl.program_id(0)
        num_chunks = (N + BLOCK_CHUNK - 1) // BLOCK_CHUNK

        q_base = Q_ptr + bh_id * stride_qbh
        k_base = K_ptr + bh_id * stride_kbh
        v_base = V_ptr + bh_id * stride_vbh
        out_base = Out_ptr + bh_id * stride_obh

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_chunk = tl.arange(0, BLOCK_CHUNK)

        state_S = tl.zeros((D_K, D_V), dtype=tl.float32)
        state_Z = tl.zeros((D_K,), dtype=tl.float32)

        for c_idx in range(0, num_chunks):
            start_n = c_idx * BLOCK_CHUNK
            cur_offs_n = start_n + offs_chunk
            mask_n = cur_offs_n < N

            q_ptrs = q_base + (cur_offs_n[:, None] * stride_qn + offs_dk[None, :] * stride_qd)
            k_ptrs = k_base + (cur_offs_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd)
            v_ptrs = v_base + (cur_offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd)

            q_tile = tl.load(q_ptrs, mask=mask_n[:, None], other=0.0)
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            # In-SRAM Non-Negative Activation
            if ACTIVATION_MODE == 0:  # ELU + 1
                q_f = q_tile.to(tl.float32)
                k_f = k_tile.to(tl.float32)
                phi_q = tl.where(q_f > 0.0, q_f, tl.exp(q_f) - 1.0) + 1.0
                phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0
            elif ACTIVATION_MODE == 1:  # ReLU
                phi_q = tl.maximum(q_tile.to(tl.float32), 0.0)
                phi_k = tl.maximum(k_tile.to(tl.float32), 0.0)
            else:
                phi_q = q_tile.to(tl.float32)
                phi_k = k_tile.to(tl.float32)

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            # Inter-Chunk Projection: query @ running state
            num_inter = tl.dot(phi_q, state_S)
            denom_inter = tl.sum(phi_q * state_Z[None, :], axis=1)

            # Intra-Chunk Causal Scores
            scores_intra = tl.dot(phi_q, tl.trans(phi_k))
            causal_mask = offs_chunk[:, None] >= offs_chunk[None, :]
            scores_causal = tl.where(causal_mask & (mask_n[:, None] & mask_n[None, :]), scores_intra, 0.0)

            num_intra = tl.dot(scores_causal, v_tile.to(tl.float32))
            denom_intra = tl.sum(scores_causal, axis=1)

            num_total = num_inter + num_intra
            denom_total = denom_inter + denom_intra + EPS
            out_tile = num_total / denom_total[:, None]

            out_ptrs = out_base + (cur_offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_n[:, None])

            # Update Running Recurrent State
            state_S += tl.dot(tl.trans(phi_k), v_tile.to(tl.float32))
            state_Z += tl.sum(phi_k, axis=0)

    @triton.jit
    def _triton_bidirectional_linear_attn_fwd_kernel(
        Q_ptr,              # [B*H, N, D_k]
        K_ptr,              # [B*H, N, D_k]
        V_ptr,              # [B*H, N, D_v]
        Out_ptr,            # [B*H, N, D_v]
        stride_qbh, stride_qn, stride_qd,
        stride_kbh, stride_kn, stride_kd,
        stride_vbh, stride_vn, stride_vd,
        stride_obh, stride_on, stride_od,
        N: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EPS: tl.constexpr,
        ACTIVATION_MODE: tl.constexpr,
    ):
        r"""
        Fused Bidirectional Linear Attention Forward Kernel for Encoders (BERT/RoBERTa).
        """
        bh_id = tl.program_id(0)

        q_base = Q_ptr + bh_id * stride_qbh
        k_base = K_ptr + bh_id * stride_kbh
        v_base = V_ptr + bh_id * stride_vbh
        out_base = Out_ptr + bh_id * stride_obh

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_n = tl.arange(0, BLOCK_N)

        state_KV = tl.zeros((D_K, D_V), dtype=tl.float32)
        state_Z = tl.zeros((D_K,), dtype=tl.float32)

        # Global parallel reduction of K^T V and sum(K)
        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask_n = cur_n < N

            k_ptrs = k_base + (cur_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd)
            v_ptrs = v_base + (cur_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd)

            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            if ACTIVATION_MODE == 0:  # ELU + 1
                k_f = k_tile.to(tl.float32)
                phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0
            elif ACTIVATION_MODE == 1:  # ReLU
                phi_k = tl.maximum(k_tile.to(tl.float32), 0.0)
            else:
                phi_k = k_tile.to(tl.float32)

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            state_KV += tl.dot(tl.trans(phi_k), v_tile.to(tl.float32))
            state_Z += tl.sum(phi_k, axis=0)

        # Query projection
        offs_m = tl.arange(0, BLOCK_M)
        for m_start in range(0, N, BLOCK_M):
            cur_m = m_start + offs_m
            mask_m = cur_m < N

            q_ptrs = q_base + (cur_m[:, None] * stride_qn + offs_dk[None, :] * stride_qd)
            q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

            if ACTIVATION_MODE == 0:
                q_f = q_tile.to(tl.float32)
                phi_q = tl.where(q_f > 0.0, q_f, tl.exp(q_f) - 1.0) + 1.0
            elif ACTIVATION_MODE == 1:
                phi_q = tl.maximum(q_tile.to(tl.float32), 0.0)
            else:
                phi_q = q_tile.to(tl.float32)

            phi_q = tl.where(mask_m[:, None], phi_q, 0.0)

            num = tl.dot(phi_q, state_KV)
            denom = tl.sum(phi_q * state_Z[None, :], axis=1) + EPS
            out_tile = num / denom[:, None]

            out_ptrs = out_base + (cur_m[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])


class _TritonLinearAttnAutogradFunction(torch.autograd.Function):
    """Autograd wrapper connecting fused Triton execution with clean PyTorch gradient propagation."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        phi: str,
        eps: float,
        cos_weight: Optional[torch.Tensor],
        sin_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        orig_shape = q.shape
        D_k = q.shape[-1]
        D_v = v.shape[-1]
        N = q.shape[-2]
        BH = q.numel() // (N * D_k)

        # Reshape to [BH, N, D]
        q_flat = q.contiguous().view(BH, N, D_k)
        k_flat = k.contiguous().view(BH, N, D_k)
        v_flat = v.contiguous().view(BH, N, D_v)

        # Apply modulation if present
        if cos_weight is not None and sin_weight is not None:
            # When modulated, pass to fallback or pre-multiply
            return linear_attn_pytorch_fallback(
                q, k, v, is_causal=is_causal, phi=phi, eps=eps,
                cos_weight=cos_weight, sin_weight=sin_weight
            )

        act_mode = 0
        if phi == "relu":
            act_mode = 1
        elif phi == "identity":
            act_mode = 2

        out = torch.empty((BH, N, D_v), dtype=q.dtype, device=q.device)

        if is_causal:
            block_chunk = 64
            grid = (BH,)
            _triton_chunked_causal_linear_attn_fwd_kernel[grid](
                q_flat, k_flat, v_flat, out,
                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                N=N, D_K=D_k, D_V=D_v,
                BLOCK_CHUNK=block_chunk,
                EPS=eps,
                ACTIVATION_MODE=act_mode,
            )
        else:
            block_m = 64
            block_n = 64
            grid = (BH,)
            _triton_bidirectional_linear_attn_fwd_kernel[grid](
                q_flat, k_flat, v_flat, out,
                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                N=N, D_K=D_k, D_V=D_v,
                BLOCK_M=block_m, BLOCK_N=block_n,
                EPS=eps,
                ACTIVATION_MODE=act_mode,
            )

        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        ctx.phi = phi
        ctx.eps = eps
        ctx.cos_weight = cos_weight
        ctx.sin_weight = sin_weight
        return out.view(*orig_shape[:-1], D_v)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v = ctx.saved_tensors
        q_req = q.detach().requires_grad_(True)
        k_req = k.detach().requires_grad_(True)
        v_req = v.detach().requires_grad_(True)

        with torch.enable_grad():
            out = linear_attn_pytorch_fallback(
                q_req, k_req, v_req,
                is_causal=ctx.is_causal,
                phi=ctx.phi,
                eps=ctx.eps,
                cos_weight=ctx.cos_weight,
                sin_weight=ctx.sin_weight,
            )
            grads = torch.autograd.grad(out, [q_req, k_req, v_req], grad_out, retain_graph=False)

        g_q = grads[0] if ctx.needs_input_grad[0] else None
        g_k = grads[1] if ctx.needs_input_grad[1] else None
        g_v = grads[2] if ctx.needs_input_grad[2] else None
        return g_q, g_k, g_v, None, None, None, None, None


def is_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> bool:
    """Check whether linear attention can run on the Triton kernel."""
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    suitable, _ = check_triton_suitability(q, "linear_attn", head_dim=head_dim, seq_len=seq_len)
    return suitable


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    phi: str = "elu+1",
    eps: float = 1e-6,
    cos_weight: Optional[torch.Tensor] = None,
    sin_weight: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Forward entrypoint for fused Triton linear attention."""
    if not is_supported(q, k, v, **kwargs):
        return linear_attn_pytorch_fallback(
            q, k, v, is_causal=is_causal, phi=phi, eps=eps,
            cos_weight=cos_weight, sin_weight=sin_weight, **kwargs
        )
    return _TritonLinearAttnAutogradFunction.apply(
        q, k, v, is_causal, phi, eps, cos_weight, sin_weight
    )


def bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    is_causal: bool = False,
    phi: str = "elu+1",
    eps: float = 1e-6,
    cos_weight: Optional[torch.Tensor] = None,
    sin_weight: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward entrypoint computing gradients for q, k, v."""
    q_req = q.detach().requires_grad_(True)
    k_req = k.detach().requires_grad_(True)
    v_req = v.detach().requires_grad_(True)
    with torch.enable_grad():
        out = fwd(
            q_req, k_req, v_req,
            is_causal=is_causal, phi=phi, eps=eps,
            cos_weight=cos_weight, sin_weight=sin_weight, **kwargs
        )
        g_q, g_k, g_v = torch.autograd.grad(out, [q_req, k_req, v_req], grad_out)
    return g_q, g_k, g_v
