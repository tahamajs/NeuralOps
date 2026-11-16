#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Triton-Accelerated SVD Linear Attention

Triton GPU-accelerated implementation of SVD-factorized chunked causal linear attention.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv
from neuralops.ops.triton import triton_dispatch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _triton_chunked_causal_linear_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
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
        Each thread block processes one (batch, head) stream sequentially across chunks,
        maintaining the recurrent state matrix S \in [D_K, D_V] and Z \in [D_K] in SRAM registers.
        """
        bh_id = tl.program_id(0)
        num_chunks = (N + BLOCK_CHUNK - 1) // BLOCK_CHUNK

        q_base = Q_ptr + bh_id * stride_qh
        k_base = K_ptr + bh_id * stride_kh
        v_base = V_ptr + bh_id * stride_vh
        out_base = Out_ptr + bh_id * stride_oh

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
                phi_q = tl.maximum(q_tile, 0.0)
                phi_k = tl.maximum(k_tile, 0.0)
            else:
                phi_q = q_tile
                phi_k = k_tile

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            num_inter = tl.dot(phi_q.to(tl.float32), state_S)
            denom_inter = tl.sum(phi_q * state_Z[None, :], axis=1)

            scores_intra = tl.dot(phi_q.to(tl.float32), tl.trans(phi_k.to(tl.float32)))
            causal_mask = offs_chunk[:, None] >= offs_chunk[None, :]
            scores_causal = tl.where(causal_mask & (mask_n[:, None] & mask_n[None, :]), scores_intra, 0.0)

            num_intra = tl.dot(scores_causal.to(tl.float32), v_tile.to(tl.float32))
            denom_intra = tl.sum(scores_causal, axis=1)

            num_total = num_inter + num_intra
            denom_total = denom_inter + denom_intra + EPS
            out_tile = num_total / denom_total[:, None]

            out_ptrs = out_base + (cur_offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_n[:, None])

            state_S += tl.dot(tl.trans(phi_k.to(tl.float32)), v_tile.to(tl.float32))
            state_Z += tl.sum(phi_k, axis=0)

    @triton.jit
    def _triton_bidirectional_linear_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
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
        Fused Bidirectional Linear Attention Forward Kernel for Encoders.
        """
        bh_id = tl.program_id(0)

        q_base = Q_ptr + bh_id * stride_qh
        k_base = K_ptr + bh_id * stride_kh
        v_base = V_ptr + bh_id * stride_vh
        out_base = Out_ptr + bh_id * stride_oh

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_n = tl.arange(0, BLOCK_N)

        acc_S = tl.zeros((D_K, D_V), dtype=tl.float32)
        acc_Z = tl.zeros((D_K,), dtype=tl.float32)

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
                phi_k = tl.maximum(k_tile, 0.0)
            else:
                phi_k = k_tile

            phi_k = tl.where(mask_n[:, None], phi_k, 0.0)
            v_tile = tl.where(mask_n[:, None], v_tile, 0.0)

            acc_S += tl.dot(tl.trans(phi_k.to(tl.float32)), v_tile.to(tl.float32))
            acc_Z += tl.sum(phi_k, axis=0)

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
                phi_q = tl.maximum(q_tile, 0.0)
            else:
                phi_q = q_tile

            num = tl.dot(phi_q.to(tl.float32), acc_S)
            denom = tl.sum(phi_q * acc_Z[None, :], axis=1) + EPS

            out_tile = num / denom[:, None]
            out_ptrs = out_base + (cur_m[:, None] * stride_on + offs_dv[None, :] * stride_od)
            tl.store(out_ptrs, out_tile.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])

    @triton.jit
    def _triton_linear_attn_bwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        dO_ptr,
        dQ_ptr,
        dK_ptr,
        dV_ptr,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        _stride_dob,
        stride_doh,
        stride_don,
        stride_dod,
        stride_dqb,
        stride_dqh,
        stride_dqn,
        stride_dqd,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        stride_dvd,
        N: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EPS: tl.constexpr,
    ):
        r"""
        Analytical Backward Kernel in Triton for Bidirectional Linear Attention.
        """
        bh_id = tl.program_id(0)

        q_base = Q_ptr + bh_id * stride_qh
        k_base = K_ptr + bh_id * stride_kh
        v_base = V_ptr + bh_id * stride_vh
        do_base = dO_ptr + bh_id * stride_doh

        dq_base = dQ_ptr + bh_id * stride_dqh
        dk_base = dK_ptr + bh_id * stride_dkh
        dv_base = dV_ptr + bh_id * stride_dvh

        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)
        offs_n = tl.arange(0, BLOCK_N)

        S = tl.zeros((D_K, D_V), dtype=tl.float32)
        Z = tl.zeros((D_K,), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask = cur_n < N
            k = tl.load(
                k_base + cur_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd,
                mask=mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + cur_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
                mask=mask[:, None],
                other=0.0,
            )
            S += tl.dot(tl.trans(k.to(tl.float32)), v.to(tl.float32))
            Z += tl.sum(k, axis=0)

        dS = tl.zeros((D_K, D_V), dtype=tl.float32)
        dZ = tl.zeros((D_K,), dtype=tl.float32)

        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask = cur_n < N

            q = tl.load(
                q_base + cur_n[:, None] * stride_qn + offs_dk[None, :] * stride_qd,
                mask=mask[:, None],
                other=0.0,
            )
            do = tl.load(
                do_base + cur_n[:, None] * stride_don + offs_dv[None, :] * stride_dod,
                mask=mask[:, None],
                other=0.0,
            )

            denom = tl.sum(q * Z[None, :], axis=1) + EPS
            inv_denom = 1.0 / denom

            do_scaled = do.to(tl.float32) * inv_denom[:, None]
            num = tl.dot(q.to(tl.float32), S)
            ratio = tl.sum(num * do_scaled, axis=1) * inv_denom

            dq = tl.dot(do_scaled, tl.trans(S)) - ratio[:, None] * Z[None, :]

            dS += tl.dot(tl.trans(q.to(tl.float32)), do_scaled)
            dZ -= tl.sum(q.to(tl.float32) * ratio[:, None], axis=0)

            tl.store(
                dq_base + cur_n[:, None] * stride_dqn + offs_dk[None, :] * stride_dqd,
                dq.to(dQ_ptr.dtype.element_ty),
                mask=mask[:, None],
            )

        for n_start in range(0, N, BLOCK_N):
            cur_n = n_start + offs_n
            mask = cur_n < N

            k = tl.load(
                k_base + cur_n[:, None] * stride_kn + offs_dk[None, :] * stride_kd,
                mask=mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_base + cur_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
                mask=mask[:, None],
                other=0.0,
            )

            dv = tl.dot(k.to(tl.float32), dS)
            dk = tl.dot(v.to(tl.float32), tl.trans(dS)) + dZ[None, :]

            tl.store(
                dv_base + cur_n[:, None] * stride_dvn + offs_dv[None, :] * stride_dvd,
                dv.to(dV_ptr.dtype.element_ty),
                mask=mask[:, None],
            )
            tl.store(
                dk_base + cur_n[:, None] * stride_dkn + offs_dk[None, :] * stride_dkd,
                dk.to(dK_ptr.dtype.element_ty),
                mask=mask[:, None],
            )


class _TritonLinearAttentionAutogradFunction(torch.autograd.Function):
    """
    Autograd Function connecting Triton forward and backward kernels on CUDA GPUs,
    with an exact numerically validated PyTorch reference fallback on CPU / Mac MPS.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        activation_mode: str = "elu",
        eps: float = 1e-6,
    ) -> torch.Tensor:
        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        act_id = 0 if activation_mode == "elu" else (1 if activation_mode == "relu" else 2)

        use_triton = HAS_TRITON and q.is_cuda and D_k in (32, 64, 128) and D_v in (32, 64, 128)

        if use_triton:
            out = torch.empty((B, H, N, D_v), device=q.device, dtype=q.dtype)
            grid = (B * H,)

            if is_causal:
                block_chunk = 32
                _triton_chunked_causal_linear_attn_fwd_kernel[grid](
                    q,
                    k,
                    v,
                    out,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    out.stride(0),
                    out.stride(1),
                    out.stride(2),
                    out.stride(3),
                    N=N,
                    D_K=D_k,
                    D_V=D_v,
                    BLOCK_CHUNK=block_chunk,
                    EPS=eps,
                    ACTIVATION_MODE=act_id,
                    num_stages=2,
                    num_warps=4,
                )
            else:
                block_m = 32
                block_n = 32
                _triton_bidirectional_linear_attn_fwd_kernel[grid](
                    q,
                    k,
                    v,
                    out,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    k.stride(0),
                    k.stride(1),
                    k.stride(2),
                    k.stride(3),
                    v.stride(0),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    out.stride(0),
                    out.stride(1),
                    out.stride(2),
                    out.stride(3),
                    N=N,
                    D_K=D_k,
                    D_V=D_v,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    EPS=eps,
                    ACTIVATION_MODE=act_id,
                    num_stages=2,
                    num_warps=4,
                )
        else:
            device_type = q.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                if activation_mode == "elu":
                    phi_q = F.elu(q.float()) + 1.0
                    phi_k = F.elu(k.float()) + 1.0
                elif activation_mode == "relu":
                    phi_q = F.relu(q.float())
                    phi_k = F.relu(k.float())
                else:
                    phi_q = q.float()
                    phi_k = k.float()

                v_f = v.float()

                if is_causal:
                    chunk_size = 64
                    num_chunks = (N + chunk_size - 1) // chunk_size
                    S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
                    Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=torch.float32)
                    outputs = []

                    for c in range(num_chunks):
                        start = c * chunk_size
                        end = min(start + chunk_size, N)
                        cur_len = end - start
                        qc = phi_q[:, :, start:end, :]
                        kc = phi_k[:, :, start:end, :]
                        vc = v_f[:, :, start:end, :]

                        scores = torch.matmul(qc, kc.transpose(-1, -2))
                        causal_mask = torch.tril(torch.ones(cur_len, cur_len, device=q.device, dtype=torch.bool))
                        scores = scores.masked_fill(~causal_mask, 0.0)
                        intra_num = torch.matmul(scores, vc)
                        intra_denom = scores.sum(dim=-1, keepdim=True)

                        inter_num = torch.matmul(qc, S)
                        inter_denom = torch.matmul(qc, Z)

                        out_c = (intra_num + inter_num) / (intra_denom + inter_denom + eps)
                        outputs.append(out_c)

                        S = S + torch.matmul(kc.transpose(-1, -2), vc)
                        Z = Z + kc.sum(dim=-2, keepdim=True).transpose(-1, -2)

                    out = torch.cat(outputs, dim=-2).to(q.dtype)
                else:
                    S = torch.matmul(phi_k.transpose(-1, -2), v_f)
                    Z = phi_k.sum(dim=-2, keepdim=True)
                    num = torch.matmul(phi_q, S)
                    denom = torch.matmul(phi_q, Z.transpose(-1, -2)) + eps
                    out = (num / denom).to(q.dtype)

        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        ctx.activation_mode = activation_mode
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        q, k, v = ctx.saved_tensors
        is_causal = ctx.is_causal
        activation_mode = ctx.activation_mode
        eps = ctx.eps

        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        use_triton_bwd = (
            HAS_TRITON
            and q.is_cuda
            and not is_causal
            and activation_mode == "identity"
            and D_k in (32, 64, 128)
            and D_v in (32, 64, 128)
        )

        if use_triton_bwd:
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            grid = (B * H,)

            _triton_linear_attn_bwd_kernel[grid](
                q,
                k,
                v,
                grad_out,
                dq,
                dk,
                dv,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                grad_out.stride(0),
                grad_out.stride(1),
                grad_out.stride(2),
                grad_out.stride(3),
                dq.stride(0),
                dq.stride(1),
                dq.stride(2),
                dq.stride(3),
                dk.stride(0),
                dk.stride(1),
                dk.stride(2),
                dk.stride(3),
                dv.stride(0),
                dv.stride(1),
                dv.stride(2),
                dv.stride(3),
                N=N,
                D_K=D_k,
                D_V=D_v,
                BLOCK_N=32,
                EPS=eps,
                num_stages=2,
                num_warps=4,
            )
            return dq, dk, dv, None, None, None

        with torch.enable_grad():
            q_req = q.detach().requires_grad_(True)
            k_req = k.detach().requires_grad_(True)
            v_req = v.detach().requires_grad_(True)

            if activation_mode == "elu":
                phi_q = F.elu(q_req.float()) + 1.0
                phi_k = F.elu(k_req.float()) + 1.0
            elif activation_mode == "relu":
                phi_q = F.relu(q_req.float())
                phi_k = F.relu(k_req.float())
            else:
                phi_q = q_req.float()
                phi_k = k_req.float()

            v_f = v_req.float()

            if is_causal:
                chunk_size = 64
                num_chunks = (N + chunk_size - 1) // chunk_size
                S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
                Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=torch.float32)
                outputs = []

                for c in range(num_chunks):
                    start = c * chunk_size
                    end = min(start + chunk_size, N)
                    cur_len = end - start
                    qc = phi_q[:, :, start:end, :]
                    kc = phi_k[:, :, start:end, :]
                    vc = v_f[:, :, start:end, :]

                    scores = torch.matmul(qc, kc.transpose(-1, -2))
                    causal_mask = torch.tril(torch.ones(cur_len, cur_len, device=q.device, dtype=torch.bool))
                    scores = scores.masked_fill(~causal_mask, 0.0)
                    intra_num = torch.matmul(scores, vc)
                    intra_denom = scores.sum(dim=-1, keepdim=True)

                    inter_num = torch.matmul(qc, S)
                    inter_denom = torch.matmul(qc, Z)

                    out_c = (intra_num + inter_num) / (intra_denom + inter_denom + eps)
                    outputs.append(out_c)

                    S = S + torch.matmul(kc.transpose(-1, -2), vc)
                    Z = Z + kc.sum(dim=-2, keepdim=True).transpose(-1, -2)

                out = torch.cat(outputs, dim=-2).to(q.dtype)
            else:
                S = torch.matmul(phi_k.transpose(-1, -2), v_f)
                Z = phi_k.sum(dim=-2, keepdim=True)
                num = torch.matmul(phi_q, S)
                denom = torch.matmul(phi_q, Z.transpose(-1, -2)) + eps
                out = (num / denom).to(q.dtype)

            dq, dk, dv = torch.autograd.grad(
                outputs=out,
                inputs=[q_req, k_req, v_req],
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
            )

            return dq, dk, dv, None, None, None


def triton_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    activation_mode: str = "elu",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convenience functional wrapper for Triton Linear Attention."""
    return _TritonLinearAttentionAutogradFunction.apply(q, k, v, is_causal, activation_mode, eps)



__all__ = ["HAS_TRITON", "_TritonLinearAttentionAutogradFunction", "triton_linear_attention"]
