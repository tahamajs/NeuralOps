#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RetNet Multi-Scale Retention

Computes multi-scale retention mechanism with exponential decay matrices and dual chunked/recurrent formulations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    class _MockTriton:
        @staticmethod
        def jit(fn):
            return fn
        @staticmethod
        def autotune(*args, **kwargs):
            return lambda fn: fn
        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b
        class Config:
            def __init__(self, *args, **kwargs):
                pass
    triton = _MockTriton()
    class _MockTl:
        constexpr = int
    tl = _MockTl()


if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _retnet_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Gammas_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        B, H, L,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b = pid_bh // H
        h = pid_bh % H

        gamma = tl.load(Gammas_ptr + h)
        log_gamma = tl.log(gamma)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        mask_m = offs_m < L

        q_base = Q_ptr + b * stride_qb + h * stride_qh
        k_base = K_ptr + b * stride_kb + h * stride_kh
        v_base = V_ptr + b * stride_vb + h * stride_vh
        out_base = Out_ptr + b * stride_ob + h * stride_oh

        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        max_k_idx = tl.minimum((pid_m + 1) * BLOCK_M, L)

        for start_n in range(0, max_k_idx, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < L

            k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(k))

            diff = offs_m[:, None] - offs_n[None, :]
            causal = (diff >= 0) & mask_m[:, None] & mask_n[None, :]
            decay = tl.where(causal, tl.exp(diff * log_gamma), 0.0)

            decayed_scores = scores * decay
            acc += tl.dot(decayed_scores.to(v.dtype), v)

        tl.store(out_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, acc.to(Q_ptr.dtype.element_ty), mask=mask_m[:, None])
else:
    def _retnet_fwd_kernel(*args, **kwargs):
        pass


def triton_multi_scale_retention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation for RetNet Multi-Scale Retention."""
    B, H, L, D = q.shape
    i_idx = torch.arange(L, device=q.device).view(1, L, 1)
    j_idx = torch.arange(L, device=q.device).view(1, 1, L)
    diff = i_idx - j_idx
    causal = diff >= 0
    g = gammas.view(H, 1, 1)
    decay = torch.where(causal, g ** diff.float(), torch.zeros_like(g))
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    A = scores * decay.unsqueeze(0)
    return torch.matmul(A, v.float()).to(q.dtype)


class TritonRetNetRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gammas: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = q.shape
        is_3d = (q.ndim == 3)
        if is_3d:
            B, L, D = q.shape
            H = 1
            q_4d = q.view(B, H, L, D)
            k_4d = k.view(B, H, L, D)
            v_4d = v.view(B, H, L, D)
            g_vec = gammas.view(1) if gammas.numel() == 1 else gammas
        else:
            B, H, L, D = q.shape
            q_4d, k_4d, v_4d, g_vec = q, k, v, gammas

        if not HAS_TRITON or not q.is_cuda or D not in (16, 32, 64, 128):
            out_4d = triton_multi_scale_retention_pytorch_ref(q_4d, k_4d, v_4d, g_vec)
            ctx.save_for_backward(q_4d, k_4d, v_4d, g_vec)
            ctx.is_3d = is_3d
            ctx.orig_shape = orig_shape
            return out_4d.view(orig_shape)

        q_c = q_4d.contiguous()
        k_c = k_4d.contiguous()
        v_c = v_4d.contiguous()
        g_c = g_vec.contiguous().to(torch.float32)
        out_4d = torch.empty_like(v_c)

        grid = lambda META: (triton.cdiv(L, META["BLOCK_M"]), B * H)

        _retnet_fwd_kernel[grid](
            q_c, k_c, v_c, g_c, out_4d,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
            v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
            out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
            B, H, L,
            D=D,
        )

        ctx.save_for_backward(q_c, k_c, v_c, g_c)
        ctx.is_3d = is_3d
        ctx.orig_shape = orig_shape
        return out_4d.view(orig_shape)

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, g_vec = ctx.saved_tensors
        B, H, L, D = q.shape
        dout_c = dout.view(B, H, L, D).contiguous()
        orig_dtype = q.dtype

        i_idx = torch.arange(L, device=q.device).view(1, L, 1)
        j_idx = torch.arange(L, device=q.device).view(1, 1, L)
        diff = i_idx - j_idx
        causal = diff >= 0
        g = g_vec.view(H, 1, 1)
        decay = torch.where(causal, g ** diff.float(), torch.zeros_like(g)).unsqueeze(0)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        A = scores * decay
        dv = torch.matmul(A.transpose(-2, -1), dout_c.float())

        dA = torch.matmul(dout_c.float(), v.float().transpose(-2, -1))
        dS = dA * decay

        dq = torch.matmul(dS, k.float())
        dk = torch.matmul(dS.transpose(-2, -1), q.float())

        dg = None
        if ctx.needs_input_grad[3]:
            ddecay = torch.where(
                causal & (diff > 0),
                diff.float() * (g ** (diff.float() - 1.0)),
                torch.zeros_like(g),
            ).unsqueeze(0)
            dgamma_mat = (dA * scores) * ddecay
            dg = dgamma_mat.sum(dim=(0, 2, 3)).to(g_vec.dtype)

        dq_out = dq.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[0] else None
        dk_out = dk.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[1] else None
        dv_out = dv.to(orig_dtype).view(ctx.orig_shape) if ctx.needs_input_grad[2] else None

        return dq_out, dk_out, dv_out, dg


def triton_multi_scale_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    """Dispatched RetNet Multi-Scale Retention with full Autograd backward support."""
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, gammas) if isinstance(t, torch.Tensor)
    ):
        return TritonRetNetRetentionFunction.apply(q, k, v, gammas)

    orig_shape = q.shape
    is_3d = (q.ndim == 3)
    if is_3d:
        B, L, D = q.shape
        H = 1
        q_4d = q.view(B, H, L, D)
        k_4d = k.view(B, H, L, D)
        v_4d = v.view(B, H, L, D)
        g_vec = gammas.view(1) if gammas.numel() == 1 else gammas
    else:
        B, H, L, D = q.shape
        q_4d, k_4d, v_4d, g_vec = q, k, v, gammas

    if not HAS_TRITON or not q.is_cuda or D not in (16, 32, 64, 128):
        return triton_multi_scale_retention_pytorch_ref(q_4d, k_4d, v_4d, g_vec).view(orig_shape)

    q_c = q_4d.contiguous()
    k_c = k_4d.contiguous()
    v_c = v_4d.contiguous()
    g_c = g_vec.contiguous().to(torch.float32)
    out_4d = torch.empty_like(v_c)

    grid = lambda META: (triton.cdiv(L, META["BLOCK_M"]), B * H)

    _retnet_fwd_kernel[grid](
        q_c, k_c, v_c, g_c, out_4d,
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
        v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
        out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
        B, H, L,
        D=D,
    )

    return out_4d.view(orig_shape)


triton_retnet_retention = triton_multi_scale_retention
triton_retnet_retention_pytorch_ref = triton_multi_scale_retention_pytorch_ref

__all__ = [
    "triton_multi_scale_retention",
    "triton_retnet_retention",
    "triton_multi_scale_retention_pytorch_ref",
    "triton_retnet_retention_pytorch_ref",
    "TritonRetNetRetentionFunction",
]
