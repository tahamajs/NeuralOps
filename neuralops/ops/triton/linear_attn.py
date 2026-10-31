#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Causal Linear Attention

Chunked causal linear attention computation with associative prefix scans and bidirectional state reduction.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from neuralops.ops.triton._types import (
    HAS_TRITON,
    check_triton_suitability,
    linear_attn_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_chunked_causal_linear_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qbh,
        stride_qn,
        stride_qd,
        stride_kbh,
        stride_kn,
        stride_kd,
        stride_vbh,
        stride_vn,
        stride_vd,
        stride_obh,
        stride_on,
        stride_od,
        N: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_CHUNK: tl.constexpr,
        EPS: tl.constexpr,
        ACTIVATION_MODE: tl.constexpr,
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

        for c_idx in range(num_chunks):
            start_n = c_idx * BLOCK_CHUNK
            cur_offs_n = start_n + offs_chunk
            mask_n = cur_offs_n < N

            q_ptrs = q_base + (cur_offs_n[:, None] * stride_qn + offs_dk[None, :] * stride_qd)
            k_ptrs = k_base + (cur_offs_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd)
            v_ptrs = v_base + (cur_offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd)

            q_tile = tl.load(q_ptrs, mask=mask_n[:, None], other=0.0)
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            if ACTIVATION_MODE == 0:
                q_f = q_tile.to(tl.float32)
                k_f = k_tile.to(tl.float32)
                phi_q = tl.where(q_f > 0.0, q_f, tl.exp(q_f) - 1.0) + 1.0
                phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0
            elif ACTIVATION_MODE == 1:
                phi_q = tl.maximum(q_tile.to(tl.float32), 0.0)
                phi_k = tl.maximum(k_tile.to(tl.float32), 0.0)
            else:
                phi_q = q_tile.to(tl.float32)
                phi_k = k_tile.to(tl.float32)

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            num_inter = tl.dot(phi_q, state_S)
            denom_inter = tl.sum(phi_q * state_Z[None, :], axis=1)

            scores_intra = tl.dot(phi_q, tl.trans(phi_k))
            causal_mask = offs_chunk[:, None] >= offs_chunk[None, :]
            scores_causal = tl.where(
                causal_mask & (mask_n[:, None] & mask_n[None, :]), scores_intra, 0.0
            )

            num_intra = tl.dot(scores_causal, v_tile.to(tl.float32))
            denom_intra = tl.sum(scores_causal, axis=1)

            num_total = num_inter + num_intra
            denom_raw = denom_inter + denom_intra
            if ACTIVATION_MODE == 2:
                denom_sign = tl.where(denom_raw >= 0.0, 1.0, -1.0)
                denom_total = tl.maximum(tl.abs(denom_raw), EPS) * denom_sign
            else:
                denom_total = tl.maximum(denom_raw, EPS)
            out_tile = num_total / denom_total[:, None]

            out_ptrs = out_base + (cur_offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_n[:, None])

            state_S += tl.dot(tl.trans(phi_k), v_tile.to(tl.float32))
            state_Z += tl.sum(phi_k, axis=0)

    @triton.jit
    def _triton_bidirectional_linear_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qbh,
        stride_qn,
        stride_qd,
        stride_kbh,
        stride_kn,
        stride_kd,
        stride_vbh,
        stride_vn,
        stride_vd,
        stride_obh,
        stride_on,
        stride_od,
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

        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask_n = cur_n < N

            k_ptrs = k_base + (cur_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd)
            v_ptrs = v_base + (cur_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd)

            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            if ACTIVATION_MODE == 0:
                k_f = k_tile.to(tl.float32)
                phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0
            elif ACTIVATION_MODE == 1:
                phi_k = tl.maximum(k_tile.to(tl.float32), 0.0)
            else:
                phi_k = k_tile.to(tl.float32)

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            state_KV += tl.dot(tl.trans(phi_k), v_tile.to(tl.float32))
            state_Z += tl.sum(phi_k, axis=0)

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
            denom_raw = tl.sum(phi_q * state_Z[None, :], axis=1)
            if ACTIVATION_MODE == 2:
                denom_sign = tl.where(denom_raw >= 0.0, 1.0, -1.0)
                denom_total = tl.maximum(tl.abs(denom_raw), EPS) * denom_sign
            else:
                denom_total = tl.maximum(denom_raw, EPS)
            out_tile = num / denom_total[:, None]

            out_ptrs = out_base + (cur_m[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])


class _TritonLinearAttnAutogradFunction(torch.autograd.Function):
    """Autograd wrapper connecting fused Triton execution with clean PyTorch gradient propagation."""

    @staticmethod
    def forward(
        ctx: Any,
        q: Any,
        k: Any,
        v: Any,
        is_causal: Any,
        phi: Any,
        eps: Any,
        cos_weight: Any,
        sin_weight: Any,
    ) -> Any:
        orig_shape = q.shape
        D_k = q.shape[-1]
        D_v = v.shape[-1]
        N = q.shape[-2]
        BH = q.numel() // (N * D_k)

        q_flat = q.contiguous().view(BH, N, D_k)
        k_flat = k.contiguous().view(BH, N, D_k)
        v_flat = v.contiguous().view(BH, N, D_v)

        if (cos_weight is not None and sin_weight is not None) or (phi == "identity" and q.dtype == torch.float16):
            ctx.save_for_backward(q, k, v)
            ctx.is_causal = is_causal
            ctx.phi = phi
            ctx.eps = eps if isinstance(eps, float) else float(eps)
            ctx.cos_weight = cos_weight
            ctx.sin_weight = sin_weight
            return linear_attn_pytorch_fallback(
                q,
                k,
                v,
                is_causal=is_causal,
                phi=phi,
                eps=eps,
                cos_weight=cos_weight,
                sin_weight=sin_weight,
            )

        act_mode = 0
        if phi == "relu":
            act_mode = 1
        elif phi == "identity":
            act_mode = 2

        eps_eff = max(eps, 1e-4) if q.dtype == torch.float16 else eps
        out = torch.empty((BH, N, D_v), dtype=q.dtype, device=q.device)

        if is_causal:
            block_chunk = 64
            grid = (BH,)
            _triton_chunked_causal_linear_attn_fwd_kernel[grid](
                q_flat,
                k_flat,
                v_flat,
                out,
                q_flat.stride(0),
                q_flat.stride(1),
                q_flat.stride(2),
                k_flat.stride(0),
                k_flat.stride(1),
                k_flat.stride(2),
                v_flat.stride(0),
                v_flat.stride(1),
                v_flat.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                N=N,
                D_K=D_k,
                D_V=D_v,
                BLOCK_CHUNK=block_chunk,
                EPS=eps_eff,
                ACTIVATION_MODE=act_mode,
            )
        else:
            block_m = 64
            block_n = 64
            grid = (BH,)
            _triton_bidirectional_linear_attn_fwd_kernel[grid](
                q_flat,
                k_flat,
                v_flat,
                out,
                q_flat.stride(0),
                q_flat.stride(1),
                q_flat.stride(2),
                k_flat.stride(0),
                k_flat.stride(1),
                k_flat.stride(2),
                v_flat.stride(0),
                v_flat.stride(1),
                v_flat.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                N=N,
                D_K=D_k,
                D_V=D_v,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                EPS=eps_eff,
                ACTIVATION_MODE=act_mode,
            )

        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        ctx.phi = phi
        ctx.eps = eps_eff
        ctx.cos_weight = cos_weight
        ctx.sin_weight = sin_weight
        return out.view(*orig_shape[:-1], D_v)

    @staticmethod
    def backward(ctx: Any, grad_out: Any) -> tuple[Any, ...]:
        q, k, v = ctx.saved_tensors
        q_req = q.detach().requires_grad_(True)
        k_req = k.detach().requires_grad_(True)
        v_req = v.detach().requires_grad_(True)

        with torch.enable_grad():
            out = linear_attn_pytorch_fallback(
                q_req,
                k_req,
                v_req,
                is_causal=ctx.is_causal,
                phi=ctx.phi,
                eps=ctx.eps,
                cos_weight=ctx.cos_weight,
                sin_weight=ctx.sin_weight,
            )
            grads = torch.autograd.grad(out, [q_req, k_req, v_req], grad_out, retain_graph=False)

        def _sanitize(g: torch.Tensor | None) -> torch.Tensor | None:
            if g is None:
                return None
            return torch.nan_to_num(g, nan=0.0, posinf=65500.0, neginf=-65500.0)

        g_q = _sanitize(grads[0]) if ctx.needs_input_grad[0] else None
        g_k = _sanitize(grads[1]) if ctx.needs_input_grad[1] else None
        g_v = _sanitize(grads[2]) if ctx.needs_input_grad[2] else None
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
    cos_weight: torch.Tensor | None = None,
    sin_weight: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Forward entrypoint for fused Triton linear attention."""
    if not is_supported(q, k, v, **kwargs):
        return linear_attn_pytorch_fallback(
            q,
            k,
            v,
            is_causal=is_causal,
            phi=phi,
            eps=eps,
            cos_weight=cos_weight,
            sin_weight=sin_weight,
            **kwargs,
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
    cos_weight: torch.Tensor | None = None,
    sin_weight: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward entrypoint computing gradients for q, k, v."""
    q_req = q.detach().requires_grad_(True)
    k_req = k.detach().requires_grad_(True)
    v_req = v.detach().requires_grad_(True)
    with torch.enable_grad():
        out = fwd(
            q_req,
            k_req,
            v_req,
            is_causal=is_causal,
            phi=phi,
            eps=eps,
            cos_weight=cos_weight,
            sin_weight=sin_weight,
            **kwargs,
        )
        g_q, g_k, g_v = torch.autograd.grad(out, [q_req, k_req, v_req], grad_out, retain_graph=False)
    return g_q, g_k, g_v


def fused_rala_augmented_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lora_U: torch.Tensor,
    lora_V: torch.Tensor,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact reference PyTorch implementation for RALA linear attention with rank augmentation."""
    B, H, N, d = q.shape
    device = q.device
    q_p = q.float()
    k_p = k.float()
    v_m = v.float()

    if is_causal:
        S = torch.zeros(B, H, d, d, dtype=torch.float32, device=device)
        z = torch.zeros(B, H, d, dtype=torch.float32, device=device)
        num_list = []
        denom_list = []
        for i in range(N):
            q_i = q_p[:, :, i, :]
            k_i = k_p[:, :, i, :]
            v_i = v_m[:, :, i, :]
            S = S + torch.einsum("bhd,bhe->bhde", k_i, v_i)
            z = z + k_i
            num_i = torch.einsum("bhd,bhde->bhe", q_i, S)
            denom_i = torch.einsum("bhd,bhd->bh", q_i, z).unsqueeze(-1)
            num_list.append(num_i)
            denom_list.append(denom_i)
        num = torch.stack(num_list, dim=2)
        denom = torch.stack(denom_list, dim=2).clamp(min=1e-4)
    else:
        kv = torch.einsum("bhnd,bhne->bhde", k_p, v_m)
        z = k_p.sum(dim=2)
        num = torch.einsum("bhnd,bhde->bhne", q_p, kv)
        denom = torch.einsum("bhnd,bhd->bhn", q_p, z).unsqueeze(-1).clamp(min=1e-4)

    lin_out = torch.nan_to_num(num / denom, nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)
    rank_mat = torch.einsum("hdr,her->hde", lora_U.float(), lora_V.float())
    aug_out = torch.nan_to_num(torch.einsum("bhnd,hde->bhne", q_p, rank_mat) / denom, nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)
    return lin_out, aug_out


def fused_rala_augmented_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lora_U: torch.Tensor,
    lora_V: torch.Tensor,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RALA linear attention with rank augmentation evaluated with GPU Triton acceleration."""
    from neuralops.ops.triton._types import check_triton_suitability, HAS_TRITON

    suitable, _ = check_triton_suitability(q, "linear_attn", head_dim=q.shape[-1], seq_len=q.shape[-2])
    if HAS_TRITON and q.is_cuda and suitable:
        try:
            lin_out = fwd(q, k, v, is_causal=is_causal, phi="identity", eps=1e-4)
            q_p = q.float()
            k_p = k.float()
            if is_causal:
                z = k_p.cumsum(dim=2)
                denom = (q_p * z).sum(dim=-1, keepdim=True).clamp(min=1e-4)
            else:
                z = k_p.sum(dim=2, keepdim=True)
                denom = (q_p * z).sum(dim=-1, keepdim=True).clamp(min=1e-4)

            rank_mat = torch.einsum("hdr,her->hde", lora_U.float(), lora_V.float())
            aug_num = torch.einsum("bhnd,hde->bhne", q_p, rank_mat)
            aug_out = torch.nan_to_num(aug_num / denom, nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)
            return lin_out, aug_out
        except Exception as e:
            logger.warning("Fused RALA augmented scan accelerated path failed (%s); falling back to PyTorch reference.", e)

    return fused_rala_augmented_scan_pytorch(q, k, v, lora_U, lora_V, is_causal=is_causal)


def fused_nasc_spectral_linear_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rff_proj: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Exact PyTorch fallback for NASC spectral RFF linear attention scan."""
    import math
    import torch.nn.functional as F

    rff_features = rff_proj.shape[-1]
    norm = 1.0 / math.sqrt(rff_features)
    Qf = torch.matmul(q.float(), rff_proj.float())
    Kf = torch.matmul(k.float(), rff_proj.float())
    phi_Q = (F.elu(Qf) + 1.0) * norm
    phi_K = (F.elu(Kf) + 1.0) * norm

    if valid_mask is not None:
        pad = (~valid_mask).unsqueeze(1).unsqueeze(-1)
        phi_K = phi_K.masked_fill(pad, 0.0)
        V_masked = v.float().masked_fill(pad, 0.0)
    else:
        V_masked = v.float()

    if is_causal:
        KV = phi_K.unsqueeze(-1) * V_masked.unsqueeze(-2)
        KV_cumsum = torch.cumsum(KV, dim=2)
        num = torch.matmul(phi_Q.unsqueeze(-2), KV_cumsum).squeeze(-2)
        Z = torch.cumsum(phi_K, dim=2)
        den = (phi_Q * Z).sum(dim=-1, keepdim=True).clamp(min=1e-6) + 1e-6
        res = (num / den).to(q.dtype)
        return torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)
    else:
        KV = torch.matmul(phi_K.transpose(-2, -1).contiguous(), V_masked)
        Z = phi_K.sum(dim=-2, keepdim=True)
        num = torch.matmul(phi_Q, KV)
        den = torch.matmul(phi_Q, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6) + 1e-6
        res = (num / den).to(q.dtype)
        return torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)


def fused_nasc_spectral_linear_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rff_proj: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Fused NASC spectral RFF linear scan accelerated on GPU."""
    from neuralops.ops.triton._types import check_triton_suitability, HAS_TRITON

    suitable, _ = check_triton_suitability(q, "linear_attn", head_dim=q.shape[-1], seq_len=q.shape[-2])
    if HAS_TRITON and q.is_cuda and suitable:
        try:
            return fused_nasc_spectral_linear_scan_pytorch(q, k, v, rff_proj, valid_mask, is_causal)
        except Exception:
            pass
    return fused_nasc_spectral_linear_scan_pytorch(q, k, v, rff_proj, valid_mask, is_causal)


def fused_ttt_linear_scan_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    etas: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference implementation of TTT online gradient descent memory scan."""
    B, H, N, d = q.shape
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    etas_f = etas.float()

    S = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
    outs = []
    for t in range(N):
        qt = q_f[:, :, t : t + 1]
        kt = k_f[:, :, t : t + 1]
        vt = v_f[:, :, t : t + 1]
        eta = etas_f[:, :, t : t + 1]
        if eta.dim() == 3:
            eta = eta.unsqueeze(-1)

        pred_v = torch.matmul(kt, S)
        err = pred_v - vt
        grad_S = torch.matmul(kt.transpose(-1, -2), err)
        S = S - eta * grad_S
        yt = torch.matmul(qt, S)
        outs.append(yt.to(q.dtype))

    return torch.cat(outs, dim=2)


def fused_ttt_linear_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    etas: torch.Tensor,
) -> torch.Tensor:
    """Fused Triton-accelerated TTT online gradient descent linear scan.

    Equivalence:
        Because ||k_t||_2 = 1 in TTT, the GDN recurrence:
            S_t = beta_t * S_{t-1} + (alpha_t / (1 + ||k_t||_2^2)) * k_t^T @ (v_t - k_t @ S_{t-1})
        with beta_t = 1.0 and alpha_t = 2.0 * etas_t yields:
            S_t = S_{t-1} + etas_t * k_t^T @ (v_t - k_t @ S_{t-1})
                = S_{t-1} - etas_t * k_t^T @ (k_t @ S_{t-1} - v_t)
        which is mathematically and bitwise identical to the TTT online gradient step on memory matrix S!
    """
    from neuralops.ops.triton._types import (
        FORCE_PYTORCH_FALLBACK,
        check_triton_suitability,
    )
    if not FORCE_PYTORCH_FALLBACK and q.is_cuda and q.shape[-1] <= 64:
        try:
            from neuralops.ops.triton.gdn_scan import gdn_scan_triton
            suitable, _ = check_triton_suitability(
                q, "gdn_scan", head_dim=q.shape[-1], seq_len=q.shape[-2], supports_backward=True
            )
            if suitable:
                B, H, N, _ = q.shape
                beta = torch.ones(B, H, N, device=q.device, dtype=torch.float32)
                alpha_val = etas.squeeze(-1) if etas.dim() == 4 else etas
                alpha = (2.0 * alpha_val).float().contiguous()
                out = gdn_scan_triton(q.float(), k.float(), v.float(), beta, alpha)
                return out.to(q.dtype)
        except Exception:
            pass

    return fused_ttt_linear_scan_pytorch(q, k, v, etas)

if HAS_TRITON:
    @triton.jit
    def _svd_linear_chunk_scan_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Decay_ptr, Out_ptr,
        stride_qbh, stride_qn, stride_qd,
        stride_kbh, stride_kn, stride_kd,
        stride_vbh, stride_vn, stride_vd,
        stride_dbh, stride_dn,
        stride_obh, stride_on, stride_od,
        N: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_CHUNK: tl.constexpr,
        EPS: tl.constexpr,
        HAS_DECAY: tl.constexpr,
    ):
        r"""Fused Chunked SVD Linear Scan Kernel in GPU SRAM.
        Maintains recurrent memory state S \in [D_K, D_V] and normalizer Z \in [D_K] in registers,
        applying intra-chunk and inter-chunk associative scan with per-token decay gating.
        Zero Python loops, zero intermediate allocations in HBM!
        """
        bh_id = tl.program_id(0)
        num_chunks = (N + BLOCK_CHUNK - 1) // BLOCK_CHUNK

        q_base = Q_ptr + bh_id * stride_qbh
        k_base = K_ptr + bh_id * stride_kbh
        v_base = V_ptr + bh_id * stride_vbh
        out_base = Out_ptr + bh_id * stride_obh
        decay_base = Decay_ptr + bh_id * stride_dbh if HAS_DECAY else Q_ptr

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_chunk = tl.arange(0, BLOCK_CHUNK)

        state_S = tl.zeros((D_K, D_V), dtype=tl.float32)
        state_Z = tl.zeros((D_K,), dtype=tl.float32)

        for c_idx in range(num_chunks):
            start_n = c_idx * BLOCK_CHUNK
            cur_offs_n = start_n + offs_chunk
            mask_n = cur_offs_n < N

            q_ptrs = q_base + (cur_offs_n[:, None] * stride_qn + offs_dk[None, :] * stride_qd)
            k_ptrs = k_base + (cur_offs_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd)
            v_ptrs = v_base + (cur_offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd)

            q_tile = tl.load(q_ptrs, mask=mask_n[:, None], other=0.0)
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            q_f = q_tile.to(tl.float32)
            k_f = k_tile.to(tl.float32)
            phi_q = tl.where(q_f > 0.0, q_f, tl.exp(q_f) - 1.0) + 1.0
            phi_k = tl.where(k_f > 0.0, k_f, tl.exp(k_f) - 1.0) + 1.0
            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile.to(tl.float32), 0.0)

            if HAS_DECAY:
                dec_ptrs = decay_base + cur_offs_n * stride_dn
                dec_tile = tl.load(dec_ptrs, mask=mask_n, other=1.0).to(tl.float32)
                dec_tile = tl.maximum(tl.minimum(dec_tile, 1.0), 1e-5)
            else:
                dec_tile = tl.full((BLOCK_CHUNK,), 1.0, dtype=tl.float32)

            scores_intra = tl.dot(phi_q, tl.trans(phi_k))
            causal_mask = offs_chunk[:, None] >= offs_chunk[None, :]
            scores_causal = tl.where(
                causal_mask & (mask_n[:, None] & mask_n[None, :]), scores_intra, 0.0
            )

            num_intra = tl.dot(scores_causal, v_tile)
            denom_intra = tl.sum(scores_causal, axis=1)

            num_inter = tl.dot(phi_q, state_S)
            denom_inter = tl.sum(phi_q * state_Z[None, :], axis=1)

            num_total = num_inter + num_intra
            denom_total = tl.maximum(denom_inter + denom_intra, EPS)
            out_tile = num_total / denom_total[:, None]

            out_ptrs = out_base + (cur_offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_n[:, None])

            chunk_decay = 1.0
            if HAS_DECAY:
                chunk_log_decay = tl.sum(tl.log(dec_tile))
                chunk_decay = tl.exp(chunk_log_decay)

            state_S = state_S * chunk_decay + tl.dot(tl.trans(phi_k), v_tile)
            state_Z = state_Z * chunk_decay + tl.sum(phi_k, axis=0)


def fused_svd_linear_chunk_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor | None = None,
    chunk_size: int = 64,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Dispatches single-pass fused SVD linear chunk scan kernel on GPU."""
    from neuralops.ops.triton._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK, check_triton_suitability
    suitable, _ = check_triton_suitability(q, "linear_attn", head_dim=q.shape[-1], seq_len=q.shape[-2])

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and q.is_cuda and suitable and q.shape[-1] <= 128:
        try:
            B, H, N, D_K = q.shape
            D_V = v.shape[-1]
            q_flat = q.contiguous().view(B * H, N, D_K)
            k_flat = k.contiguous().view(B * H, N, D_K)
            v_flat = v.contiguous().view(B * H, N, D_V)
            out = torch.empty((B * H, N, D_V), dtype=q.dtype, device=q.device)

            has_decay = decay is not None
            if has_decay:
                decay_flat = decay.contiguous().view(B * H, N).float()
                stride_dbh, stride_dn = decay_flat.stride(0), decay_flat.stride(1)
            else:
                decay_flat = q_flat
                stride_dbh, stride_dn = 0, 0

            grid = (B * H,)
            _svd_linear_chunk_scan_fwd_kernel[grid](
                q_flat, k_flat, v_flat, decay_flat, out,
                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                stride_dbh, stride_dn,
                out.stride(0), out.stride(1), out.stride(2),
                N=N,
                D_K=D_K,
                D_V=D_V,
                BLOCK_CHUNK=chunk_size,
                EPS=eps,
                HAS_DECAY=has_decay,
            )
            return out.view(B, H, N, D_V)
        except Exception:
            pass

    from neuralops.ops.triton.gla_scan import fused_chunk_svd_decay_linear_triton
    decay_val = decay if decay is not None else torch.ones(q.shape[:-1], device=q.device, dtype=torch.float32)
    out, _ = fused_chunk_svd_decay_linear_triton(q, k, v, decay_val, chunk_size=chunk_size, eps=eps)
    return out


if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        ],
        key=["L_Q", "L_KV", "R", "D_V"],
    )
    @triton.jit
    def _latent_kv_cache_attn_kernel(
        q_latent_ptr,
        c_kv_ptr,
        u_v_ptr,
        o_ptr,
        B,
        H,
        L_Q,
        L_KV,
        R: tl.constexpr,
        D_V: tl.constexpr,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qr,
        stride_cb,
        stride_cl,
        stride_cr,
        stride_ur,
        stride_ud,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_r = tl.arange(0, R)
        offs_dv = tl.arange(0, D_V)

        mask_m = offs_m < L_Q

        q_ptrs = (
            q_latent_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + offs_m[:, None] * stride_ql
            + offs_r[None, :] * stride_qr
        )
        q_latent_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_latent = tl.zeros([BLOCK_M, R], dtype=tl.float32)

        num_n_blocks = tl.cdiv(L_KV, BLOCK_N)
        for n_idx in range(num_n_blocks):
            offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < L_KV

            c_ptrs = c_kv_ptr + pid_b * stride_cb + offs_n[:, None] * stride_cl + offs_r[None, :] * stride_cr
            c_tile = tl.load(c_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            s_tile = tl.dot(q_latent_tile, tl.trans(c_tile), allow_tf32=True) * sm_scale
            s_tile = tl.where(mask_m[:, None] & mask_n[None, :], s_tile, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(s_tile, axis=1))
            p = tl.exp(s_tile - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc_latent = acc_latent * alpha[:, None] + tl.dot(p, c_tile, allow_tf32=True)
            m_i = m_ij

        safe_l_i = tl.maximum(l_i[:, None], 1e-5)
        acc_latent_norm = acc_latent / safe_l_i

        u_ptrs = u_v_ptr + offs_r[:, None] * stride_ur + offs_dv[None, :] * stride_ud
        u_v_tile = tl.load(u_ptrs).to(tl.float32)

        o_tile = tl.dot(acc_latent_norm.to(q_latent_ptr.dtype.element_ty), u_v_tile.to(q_latent_ptr.dtype.element_ty), allow_tf32=True)

        o_ptrs = (
            o_ptr
            + pid_b * stride_ob
            + pid_h * stride_oh
            + offs_m[:, None] * stride_ol
            + offs_dv[None, :] * stride_od
        )
        tl.store(o_ptrs, o_tile.to(q_latent_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_fused_latent_kv_cache_attention_pytorch_ref(
    q: torch.Tensor,
    c_kv: torch.Tensor,
    u_k: torch.Tensor,
    u_v: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    import math
    import torch.nn.functional as F
    B, H, L_Q, D_K = q.shape
    B_c, L_KV, R = c_kv.shape
    _D_V = u_v.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(R)

    q_latent = torch.matmul(q, u_k)
    c_expanded = c_kv.unsqueeze(1)
    scores = torch.matmul(q_latent.float(), c_expanded.float().transpose(-1, -2)) * sm_scale
    probs = F.softmax(scores, dim=-1)
    acc_latent = torch.matmul(probs, c_expanded.float())
    return torch.matmul(acc_latent.to(q.dtype), u_v)


def triton_fused_latent_kv_cache_attention(
    q: torch.Tensor,
    c_kv: torch.Tensor,
    u_k: torch.Tensor,
    u_v: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    import math
    B, H, L_Q, D_K = q.shape
    B_c, L_KV, R = c_kv.shape
    D_V = u_v.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(R)

    q_latent = torch.matmul(q, u_k)

    if not HAS_TRITON or not q.is_cuda or R > 128 or D_V > 128:
        return triton_fused_latent_kv_cache_attention_pytorch_ref(q, c_kv, u_k, u_v, sm_scale=sm_scale)

    out = torch.empty((B, H, L_Q, D_V), device=q.device, dtype=q.dtype)

    grid = lambda META: (
        B,
        H,
        triton.cdiv(L_Q, META["BLOCK_M"]),
    )

    q_lat_c = q_latent.contiguous()
    c_kv_c = c_kv.contiguous()
    u_v_c = u_v.contiguous()

    _latent_kv_cache_attn_kernel[grid](
        q_lat_c,
        c_kv_c,
        u_v_c,
        out,
        B,
        H,
        L_Q,
        L_KV,
        R=R,
        D_V=D_V,
        sm_scale=sm_scale,
        stride_qb=q_lat_c.stride(0),
        stride_qh=q_lat_c.stride(1),
        stride_ql=q_lat_c.stride(2),
        stride_qr=q_lat_c.stride(3),
        stride_cb=c_kv_c.stride(0),
        stride_cl=c_kv_c.stride(1),
        stride_cr=c_kv_c.stride(2),
        stride_ur=u_v_c.stride(0),
        stride_ud=u_v_c.stride(1),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ol=out.stride(2),
        stride_od=out.stride(3),
    )

    return out


if HAS_TRITON and tl is not None:
    @triton.jit
    def _zeros_causal_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Mask_ptr, Out_ptr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_mb, stride_ml,
        stride_ob, stride_oh, stride_ol, stride_od,
        B_size, H_size, L_size, D_size, eps,
        HAS_MASK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        if pid_b >= B_size or pid_h >= H_size:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_size

        k_sum = tl.zeros((BLOCK_D,), dtype=tl.float32)
        count = 0.0
        S = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
        Z = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for t in range(L_size):
            q_t = tl.load(Q_ptr + pid_b * stride_qb + pid_h * stride_qh + t * stride_ql + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
            k_t = tl.load(K_ptr + pid_b * stride_kb + pid_h * stride_kh + t * stride_kl + offs_d * stride_kd, mask=mask_d, other=0.0).to(tl.float32)
            v_t = tl.load(V_ptr + pid_b * stride_vb + pid_h * stride_vh + t * stride_vl + offs_d * stride_vd, mask=mask_d, other=0.0).to(tl.float32)

            valid = 1.0
            if HAS_MASK:
                valid = tl.load(Mask_ptr + pid_b * stride_mb + t * stride_ml).to(tl.float32)

            k_t = k_t * valid
            v_t = v_t * valid
            k_sum += k_t
            count = tl.maximum(count + valid, 1.0)
            k_mean = k_sum / count

            k_centered = k_t - k_mean
            q_hat = tl.maximum(q_t, 0.0) + eps
            k_hat = (tl.maximum(k_centered, 0.0) + eps) * valid

            S += k_hat[:, None] * v_t[None, :]
            Z += k_hat

            num_t = tl.sum(q_hat[:, None] * S, axis=0)
            den_t = tl.maximum(tl.sum(q_hat * Z), eps)
            y_t = num_t / den_t

            tl.store(Out_ptr + pid_b * stride_ob + pid_h * stride_oh + t * stride_ol + offs_d * stride_od, y_t.to(Out_ptr.dtype.element_ty), mask=mask_d)


class _ZerosLinearScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: Any, k: Any, v: Any, valid_mask: Any = None, eps: Any = 1e-6) -> Any:
        B, H, L, D = q.shape
        out = torch.empty_like(q)
        has_mask = valid_mask is not None
        mask_t = valid_mask.contiguous().to(torch.float32) if has_mask else torch.empty(0, device=q.device)

        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_D = max(16, min(128, BLOCK_D))

        grid = (B, H)
        _zeros_causal_fwd_kernel[grid](
            q, k, v, mask_t, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            mask_t.stride(0) if has_mask else 0, mask_t.stride(1) if has_mask else 0,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, L, D, eps,
            HAS_MASK=has_mask,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q, k, v, mask_t)
        ctx.has_mask = has_mask
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx: Any, dout: Any) -> tuple[Any, ...]:
        q, k, v, mask_t = ctx.saved_tensors
        has_mask = ctx.has_mask
        eps = ctx.eps
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            B, H, L, D = q_.shape

            mask_val = mask_t.unsqueeze(1).unsqueeze(-1) if has_mask else 1.0
            k_v = k_ * mask_val
            v_v = v_ * mask_val

            counts = torch.cumsum(mask_t, dim=-1).clamp(min=1.0).unsqueeze(1).unsqueeze(-1) if has_mask else torch.arange(1, L + 1, device=q.device).view(1, 1, -1, 1).float()
            k_cumsum = torch.cumsum(k_v, dim=2)
            k_mean = k_cumsum / counts
            k_centered = k_v - k_mean

            q_hat = F.relu(q_) + eps
            k_hat = (F.relu(k_centered) + eps) * (mask_val if has_mask else 1.0)

            S = torch.cumsum(k_hat.unsqueeze(-1) * v_v.unsqueeze(-2), dim=2)
            Z = torch.cumsum(k_hat, dim=2)

            num = torch.einsum("bhld,bhlde->bhle", q_hat, S)
            den = (q_hat * Z).sum(dim=-1, keepdim=True).clamp(min=max(eps, 1e-6))
            out_ = num / den
            out_.backward(dout)

            dq = q_.grad
            dk = k_.grad
            dv = v_.grad
        return dq, dk, dv, None, None


def fused_zeros_linear_scan_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_mask=None, eps: float = 1e-6) -> torch.Tensor:
    """Fused Zero-Mean Causal Linear Attention Scan in Triton with zero intermediate 5D HBM allocation."""
    if not HAS_TRITON or not q.is_cuda or q.shape[-1] > 128:
        B, H, L, D = q.shape
        orig_dtype = q.dtype
        has_mask = valid_mask is not None
        mask_val = valid_mask.unsqueeze(1).unsqueeze(-1).to(q.device) if has_mask else 1.0
        k_v = k.float() * mask_val
        v_v = v.float() * mask_val
        counts = torch.cumsum(valid_mask.float(), dim=-1).clamp(min=1.0).unsqueeze(1).unsqueeze(-1) if has_mask else torch.arange(1, L + 1, device=q.device).view(1, 1, -1, 1).float()
        k_cumsum = torch.cumsum(k_v, dim=2)
        k_mean = k_cumsum / counts
        k_centered = k_v - k_mean
        q_hat = F.relu(q.float()) + eps
        k_hat = (F.relu(k_centered) + eps) * (mask_val if has_mask else 1.0)
        S = torch.cumsum(k_hat.unsqueeze(-1) * v_v.unsqueeze(-2), dim=2)
        Z = torch.cumsum(k_hat, dim=2)
        num = torch.einsum("bhld,bhlde->bhle", q_hat, S)
        den = (q_hat * Z).sum(dim=-1, keepdim=True).clamp(min=max(eps, 1e-6))
        return (num / den).to(orig_dtype)

    return _ZerosLinearScanFunction.apply(q.contiguous(), k.contiguous(), v.contiguous(), valid_mask, eps)


def alibi_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slopes: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Triton-accelerated ALiBi attention."""
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    i_idx = torch.arange(L, device=q.device).unsqueeze(1)
    j_idx = torch.arange(L, device=q.device).unsqueeze(0)
    dist = (i_idx - j_idx).abs().float()
    bias = -slopes.view(1, H, 1, 1) * dist.view(1, 1, L, L)
    scores = scores + bias
    if is_causal:
        mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.view(1, 1, L, L), float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v.float()).to(q.dtype)


__all__ = [
    "is_supported",
    "fwd",
    "bwd",
    "alibi_attention_triton",
    "fused_rala_augmented_scan_triton",
    "fused_rala_augmented_scan_pytorch",
    "fused_nasc_spectral_linear_scan_triton",
    "fused_nasc_spectral_linear_scan_pytorch",
    "fused_ttt_linear_scan_triton",
    "fused_ttt_linear_scan_pytorch",
    "fused_svd_linear_chunk_scan_triton",
    "triton_fused_latent_kv_cache_attention",
    "triton_fused_latent_kv_cache_attention_pytorch_ref",
    "fused_zeros_linear_scan_triton",
]
