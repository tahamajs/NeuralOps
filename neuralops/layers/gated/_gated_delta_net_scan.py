#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Scan operations and recurrence kernels for Gated DeltaNet."""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from neuralops.ops.triton.gdn_scan import MAX_REGISTER_DIM, gdn_scan_triton

    HAS_TRITON_GDN = True
    logger.info("GDN: Triton scan available")
except Exception as _e:
    logger.warning("GDN: Triton scan unavailable: %s", _e)
    gdn_scan_triton = None
    HAS_TRITON_GDN = False
    MAX_REGISTER_DIM = 64


def _delta_chunk_forward(
    S: torch.Tensor,
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    v_i: torch.Tensor,
    beta_i: torch.Tensor,
    alpha_i: torch.Tensor,
    mask_i: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel chunked WY delta-rule recurrence over one chunk of size C.

    Exact algebraic equivalent to sequential steps:
        inter_v = k_t · S_{t-1}
        err     = v_t - inter_v
        step    = alpha_t / (1 + ||k_t||²)
        S_t     = beta_t · S_{t-1} + step · k_tᵀ · err
        y_t     = q_t · S_t

    If mask_i is provided (shape (B, H, C) bool), padded steps do not update S.
    """
    C = q_i.shape[2]
    if mask_i is not None:
        m_f = mask_i.unsqueeze(-1).to(k_i.dtype)
        k_m = k_i * m_f
        v_m = v_i * m_f
    else:
        k_m = k_i
        v_m = v_i

    dev_type = S.device.type
    ctx = (
        torch.amp.autocast(device_type=dev_type, enabled=False)
        if hasattr(torch, "amp") and dev_type in ("cuda", "cpu")
        else contextlib.nullcontext()
    )
    with ctx:
        S_f = S.float()
        q_f = q_i.float()
        k_m = k_m.float()
        v_m = v_m.float()
        beta_f = beta_i.float().clamp(min=1e-6)
        alpha_f = alpha_i.float()

        norm_k_sq = (k_m * k_m).sum(dim=-1)
        w = alpha_f / (1.0 + norm_k_sq).clamp_min(1.0)

        if C == 1:
            inter_v = torch.matmul(k_m, S_f)
            err = v_m - inter_v
            delta = w.unsqueeze(-1) * torch.matmul(k_m.transpose(-1, -2), err)
            S_next = beta_f.unsqueeze(-1) * S_f + delta
            S_next = torch.nan_to_num(S_next, nan=0.0, posinf=1e4, neginf=-1e4)
            y_i = torch.matmul(q_f, S_next)
            return y_i, S_next

        log_beta = torch.log(beta_f)
        gamma = torch.cumsum(log_beta, dim=-1)
        gamma_prev = gamma - log_beta

        tril_strict = torch.tril(
            torch.ones((C, C), device=S_f.device, dtype=S_f.dtype),
            diagonal=-1,
        )
        tril_incl = torch.tril(
            torch.ones((C, C), device=S_f.device, dtype=S_f.dtype),
            diagonal=0,
        )

        # Clamp exponent difference to <= 0 so the masked upper triangle never overflows exp().
        decay_m = torch.exp((gamma_prev.unsqueeze(-1) - gamma.unsqueeze(-2)).clamp(max=0.0)) * tril_strict
        M = torch.matmul(k_m, k_m.transpose(-1, -2)) * (w.unsqueeze(-1) * decay_m)
        eye = torch.eye(C, device=S_f.device, dtype=S_f.dtype)
        I_M = eye + M

        rhs = w.unsqueeze(-1) * (v_m - torch.exp(gamma_prev).unsqueeze(-1) * torch.matmul(k_m, S_f))
        U = torch.linalg.solve_triangular(I_M, rhs, upper=False, unitriangular=True)

        decay_end = torch.exp(gamma[:, :, -1])
        scale_to_end = torch.exp((gamma[:, :, -1:] - gamma).clamp(max=0.0))
        k_scaled = k_m * scale_to_end.unsqueeze(-1)
        S_next = decay_end.unsqueeze(-1).unsqueeze(-1) * S_f + torch.matmul(k_scaled.transpose(-1, -2), U)
        S_next = torch.nan_to_num(S_next, nan=0.0, posinf=1e4, neginf=-1e4)

        decay_qk = torch.exp((gamma.unsqueeze(-1) - gamma.unsqueeze(-2)).clamp(max=0.0)) * tril_incl
        attn_qk = torch.matmul(q_f, k_m.transpose(-1, -2)) * decay_qk
        y_i = torch.exp(gamma).unsqueeze(-1) * torch.matmul(q_f, S_f) + torch.matmul(attn_qk, U)
        return y_i, S_next


def _clamp_fp16_safe_to_dtype(y: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Sanitize non-finite values and clamp within FP16 safe bounds before casting."""
    fp16_lim = torch.finfo(torch.float16).max * 0.5
    y = torch.nan_to_num(y, nan=0.0, posinf=fp16_lim, neginf=-fp16_lim)
    if target_dtype.is_floating_point:
        lim = min(fp16_lim, torch.finfo(target_dtype).max * 0.5)
        y = y.clamp(min=-lim, max=lim)
    return y.to(target_dtype)


def chunked_delta_rule_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked Delta-Rule scan.

    Args:
        q, k, v: (B, H, L, d)
        beta:    (B, H, L)  — per-head decay in (0, 1)
        alpha:   (B, H, L)  — per-head step size in (0, 1)
        initial_state: (B, H, d, d) or None
        mask:    (B, L) bool or None  — True for valid tokens
        chunk_size: int

    Returns:
        y:  (B, H, L, d)
        S:  (B, H, d, d)  — final recurrent state
    """
    B, H, L, d = q.shape
    device = q.device

    q_f = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
    k_f = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
    v_f = v.float()
    beta_f = beta.float().clamp(min=1e-4, max=1.0 - 1e-4)
    alpha_f = alpha.float().clamp(min=1e-4, max=1.0)

    mask_h = None
    if mask is not None:
        mask_h = mask[:, None, :].expand(-1, H, -1).contiguous()

    align_size = (
        64
        if chunk_size <= 64 and 64 % chunk_size == 0
        else (chunk_size if chunk_size % 64 == 0 else math.lcm(64, chunk_size))
    )
    pad_len = (align_size - (L % align_size)) % align_size if L > 1 else 0
    if pad_len > 0:
        q_f = F.pad(q_f, (0, 0, 0, pad_len))
        k_f = F.pad(k_f, (0, 0, 0, pad_len))
        v_f = F.pad(v_f, (0, 0, 0, pad_len))
        beta_f = F.pad(beta_f, (0, pad_len), value=1.0)
        alpha_f = F.pad(alpha_f, (0, pad_len), value=0.0)
        if mask_h is not None:
            mask_h = F.pad(mask_h, (0, pad_len), value=False)

    if (
        HAS_TRITON_GDN
        and gdn_scan_triton is not None
        and q.is_cuda
        and d <= MAX_REGISTER_DIM
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        try:
            out, S_final = gdn_scan_triton(
                q_f,
                k_f,
                v_f,
                beta_f,
                alpha_f,
                initial_state=initial_state,
                mask=mask_h,
                return_final_state=True,
                normalize_qk=False,
            )
            if pad_len > 0:
                out = out[:, :, :L, :]
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)
            return out, S_final.to(device=device, dtype=torch.float32)
        except Exception as e:
            logger.warning("Triton gdn_scan failed, falling back to PyTorch: %s", e)

    if (
        q.is_cuda
        and d <= 64
        and initial_state is None
        and mask_h is None
        and not (q.requires_grad or k.requires_grad or v.requires_grad)
    ):
        try:
            from fla.ops import chunk_gdn

            q_fl = q_f.to(q.dtype).transpose(1, 2).contiguous()
            k_fl = k_f.to(q.dtype).transpose(1, 2).contiguous()
            v_fl = v_f.to(q.dtype).transpose(1, 2).contiguous()
            log_g = torch.log(beta_f.clamp(min=1e-4, max=1.0)).transpose(1, 2).to(q.dtype).contiguous()
            step_beta = alpha_f.transpose(1, 2).to(q.dtype).contiguous()
            res = chunk_gdn(q_fl, k_fl, v_fl, log_g, step_beta)
            out_fl = res[0] if isinstance(res, tuple) else res
            out = out_fl.transpose(1, 2).contiguous()
            if pad_len > 0:
                out = out[:, :, :L, :]
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)
            S_final = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
            return out, S_final
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

    if initial_state is not None:
        S = initial_state.to(device=device, dtype=torch.float32).clone()
    else:
        S = torch.zeros((B, H, d, d), device=device, dtype=torch.float32)

    if L == 1:
        y_i, S_out = _delta_chunk_forward(S, q_f, k_f, v_f, beta_f, alpha_f, mask_h)
        return _clamp_fp16_safe_to_dtype(y_i, q.dtype), S_out

    L_pad = L + pad_len
    n_chunks = L_pad // chunk_size

    q_c = q_f.view(B, H, n_chunks, chunk_size, d)
    k_c = k_f.view(B, H, n_chunks, chunk_size, d)
    v_c = v_f.view(B, H, n_chunks, chunk_size, d)
    beta_c = beta_f.view(B, H, n_chunks, chunk_size)
    alpha_c = alpha_f.view(B, H, n_chunks, chunk_size)
    mask_c = mask_h.view(B, H, n_chunks, chunk_size) if mask_h is not None else None

    outputs = []
    use_cp = torch.is_grad_enabled()
    if use_cp:
        import torch.utils.checkpoint as cp

    for c in range(n_chunks):
        mk = mask_c[:, :, c] if mask_c is not None else None
        if use_cp:
            y_i, S = cp.checkpoint(
                _delta_chunk_forward,
                S,
                q_c[:, :, c],
                k_c[:, :, c],
                v_c[:, :, c],
                beta_c[:, :, c],
                alpha_c[:, :, c],
                mk,
                use_reentrant=False,
            )
        else:
            y_i, S = _delta_chunk_forward(
                S,
                q_c[:, :, c],
                k_c[:, :, c],
                v_c[:, :, c],
                beta_c[:, :, c],
                alpha_c[:, :, c],
                mk,
            )
        outputs.append(y_i)

    y = torch.cat(outputs, dim=2)
    if pad_len > 0:
        y = y[:, :, :L, :]

    return _clamp_fp16_safe_to_dtype(y, q.dtype), S
