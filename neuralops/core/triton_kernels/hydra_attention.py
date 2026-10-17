"""
Triton JIT Hardware Kernel for Hydra Attention (v1.4.0 Flagship).
Target Architecture: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
Mathematical formulation:
  S = (1 / L) * sum_{t=1}^L (K_t * V_t)       [Global channel-wise state aggregation]
  Y_t = Q_t * S                               [Linear token modulation]
Exact analytical autograd backward:
  dQ_t = dY_t * S
  T = sum_{t=1}^L (dY_t * Q_t)
  dK_t = (1 / L) * T * V_t
  dV_t = (1 / L) * T * K_t
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
            triton.Config({"BLOCK_L": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_L": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_L": 128, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_L": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _hydra_fwd_kernel(
        Q_ptr, S_ptr, Out_ptr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_sb, stride_sh, stride_sd,
        stride_ob, stride_oh, stride_ol, stride_od,
        B, H, L,
        D: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_l = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b = pid_bh // H
        h = pid_bh % H

        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_d = tl.arange(0, D)

        mask_l = offs_l < L

        s = tl.load(S_ptr + b * stride_sb + h * stride_sh + offs_d * stride_sd)
        q = tl.load(
            Q_ptr + b * stride_qb + h * stride_qh + offs_l[:, None] * stride_ql + offs_d[None, :] * stride_qd,
            mask=mask_l[:, None],
            other=0.0,
        )

        out = q * s[None, :]

        tl.store(
            Out_ptr + b * stride_ob + h * stride_oh + offs_l[:, None] * stride_ol + offs_d[None, :] * stride_od,
            out,
            mask=mask_l[:, None],
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_L": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_L": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_L": 256}, num_warps=4, num_stages=3),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _hydra_bwd_kernel(
        Dout_ptr, Q_ptr, K_ptr, V_ptr, S_ptr, T_ptr,
        Dq_ptr, Dk_ptr, Dv_ptr,
        stride_dout_b, stride_dout_h, stride_dout_l, stride_dout_d,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_sb, stride_sh, stride_sd,
        stride_tb, stride_th, stride_td,
        stride_dq_b, stride_dq_h, stride_dq_l, stride_dq_d,
        stride_dk_b, stride_dk_h, stride_dk_l, stride_dk_d,
        stride_dv_b, stride_dv_h, stride_dv_l, stride_dv_d,
        inv_L,
        B, H, L,
        D: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        pid_l = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b = pid_bh // H
        h = pid_bh % H

        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_d = tl.arange(0, D)

        mask_l = offs_l < L

        s = tl.load(S_ptr + b * stride_sb + h * stride_sh + offs_d * stride_sd)
        t = tl.load(T_ptr + b * stride_tb + h * stride_th + offs_d * stride_td)

        dout = tl.load(
            Dout_ptr + b * stride_dout_b + h * stride_dout_h + offs_l[:, None] * stride_dout_l + offs_d[None, :] * stride_dout_d,
            mask=mask_l[:, None],
            other=0.0,
        )
        k = tl.load(
            K_ptr + b * stride_kb + h * stride_kh + offs_l[:, None] * stride_kl + offs_d[None, :] * stride_kd,
            mask=mask_l[:, None],
            other=0.0,
        )
        v = tl.load(
            V_ptr + b * stride_vb + h * stride_vh + offs_l[:, None] * stride_vl + offs_d[None, :] * stride_vd,
            mask=mask_l[:, None],
            other=0.0,
        )

        dq = dout * s[None, :]
        dk = (t[None, :] * inv_L) * v
        dv = (t[None, :] * inv_L) * k

        tl.store(
            Dq_ptr + b * stride_dq_b + h * stride_dq_h + offs_l[:, None] * stride_dq_l + offs_d[None, :] * stride_dq_d,
            dq,
            mask=mask_l[:, None],
        )
        tl.store(
            Dk_ptr + b * stride_dk_b + h * stride_dk_h + offs_l[:, None] * stride_dk_l + offs_d[None, :] * stride_dk_d,
            dk,
            mask=mask_l[:, None],
        )
        tl.store(
            Dv_ptr + b * stride_dv_b + h * stride_dv_h + offs_l[:, None] * stride_dv_l + offs_d[None, :] * stride_dv_d,
            dv,
            mask=mask_l[:, None],
        )
else:
    def _hydra_fwd_kernel(*args, **kwargs): pass
    def _hydra_bwd_kernel(*args, **kwargs): pass


def triton_hydra_attention_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Exact PyTorch reference implementation for Hydra Attention."""
    global_kv = (k.float() * v.float()).mean(dim=-2, keepdim=True)
    return (q.float() * global_kv).to(q.dtype)


class TritonHydraAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = q.shape
        is_3d = (q.ndim == 3)
        if is_3d:
            B, L, D = q.shape
            H = 1
            q_4d = q.view(B, H, L, D)
            k_4d = k.view(B, H, L, D)
            v_4d = v.view(B, H, L, D)
        else:
            B, H, L, D = q.shape
            q_4d, k_4d, v_4d = q, k, v

        if not HAS_TRITON or not q.is_cuda or D not in (16, 32, 64, 128, 256):
            out_4d = triton_hydra_attention_pytorch_ref(q_4d, k_4d, v_4d)
            s_vec = (k_4d.float() * v_4d.float()).mean(dim=2).to(q.dtype)
            ctx.save_for_backward(q_4d, k_4d, v_4d, s_vec)
            ctx.is_3d = is_3d
            ctx.orig_shape = orig_shape
            return out_4d.view(orig_shape)

        q_c = q_4d.contiguous()
        k_c = k_4d.contiguous()
        v_c = v_4d.contiguous()

        # Compute per-head global KV state: S [B, H, D]
        s_vec = (k_c.float() * v_c.float()).mean(dim=2).to(q_c.dtype).contiguous()
        out_4d = torch.empty_like(q_c)

        grid = lambda META: (triton.cdiv(L, META["BLOCK_L"]), B * H)

        _hydra_fwd_kernel[grid](
            q_c, s_vec, out_4d,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            s_vec.stride(0), s_vec.stride(1), s_vec.stride(2),
            out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
            B, H, L,
            D=D,
        )

        ctx.save_for_backward(q_c, k_c, v_c, s_vec)
        ctx.is_3d = is_3d
        ctx.orig_shape = orig_shape
        return out_4d.view(orig_shape)

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        q, k, v, s_vec = ctx.saved_tensors
        B, H, L, D = q.shape
        dout_c = dout.view(B, H, L, D).contiguous()

        # T = sum_{t=1}^L (dout * q) [B, H, D]
        t_vec = (dout_c.float() * q.float()).sum(dim=2).to(dout_c.dtype).contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid = lambda META: (triton.cdiv(L, META["BLOCK_L"]), B * H)

        _hydra_bwd_kernel[grid](
            dout_c, q, k, v, s_vec, t_vec,
            dq, dk, dv,
            dout_c.stride(0), dout_c.stride(1), dout_c.stride(2), dout_c.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            s_vec.stride(0), s_vec.stride(1), s_vec.stride(2),
            t_vec.stride(0), t_vec.stride(1), t_vec.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            1.0 / float(L),
            B, H, L,
            D=D,
        )

        dq_out = dq.view(ctx.orig_shape) if ctx.needs_input_grad[0] else None
        dk_out = dk.view(ctx.orig_shape) if ctx.needs_input_grad[1] else None
        dv_out = dv.view(ctx.orig_shape) if ctx.needs_input_grad[2] else None

        return dq_out, dk_out, dv_out


def triton_hydra_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Dispatched Hydra Attention with full Autograd backward support."""
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)
    ):
        return TritonHydraAttentionFunction.apply(q, k, v)

    orig_shape = q.shape
    is_3d = (q.ndim == 3)
    if is_3d:
        B, L, D = q.shape
        H = 1
        q_4d = q.view(B, H, L, D)
        k_4d = k.view(B, H, L, D)
        v_4d = v.view(B, H, L, D)
    else:
        B, H, L, D = q.shape
        q_4d, k_4d, v_4d = q, k, v

    if not HAS_TRITON or not q.is_cuda or D not in (16, 32, 64, 128, 256):
        return triton_hydra_attention_pytorch_ref(q, k, v)

    q_c = q_4d.contiguous()
    k_c = k_4d.contiguous()
    v_c = v_4d.contiguous()

    s_vec = (k_c.float() * v_c.float()).mean(dim=2).to(q_c.dtype).contiguous()
    out_4d = torch.empty_like(q_c)

    grid = lambda META: (triton.cdiv(L, META["BLOCK_L"]), B * H)

    _hydra_fwd_kernel[grid](
        q_c, s_vec, out_4d,
        q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
        s_vec.stride(0), s_vec.stride(1), s_vec.stride(2),
        out_4d.stride(0), out_4d.stride(1), out_4d.stride(2), out_4d.stride(3),
        B, H, L,
        D=D,
    )

    return out_4d.view(orig_shape)


__all__ = [
    "triton_hydra_attention",
    "triton_hydra_attention_pytorch_ref",
    "TritonHydraAttentionFunction",
]
